/*
 * gating.js — client-side progress gating for the "Build an SLM From Scratch"
 * course (content-core, browser modules 0-6 + the S1 tokenizer warm-up).
 *
 * WHAT THIS IS, AND WHAT IT IS NOT
 * --------------------------------
 * This file records *per-browser, per-device* lesson-completion flags so a
 * learner can close the tab and come back without re-running everything. It is
 * convenience state, NOT a system of record. Per the design spec (§3.4):
 *
 *   - IndexedDB (with a localStorage fallback) is best-effort scratch state.
 *     It is not synced across devices and can be wiped by cache-clear or
 *     browser storage eviction.
 *   - The durable backup is an exported JSON "progress token" the learner
 *     downloads and can paste back to restore. That token — not IndexedDB —
 *     is the thing a learner should keep.
 *   - All browser-side gates are honor-system with ZERO integrity guarantee.
 *     A future server-side LMS must re-grade and must NOT import these flags
 *     as trusted records.
 *
 * PUBLIC API (kept deliberately small):
 *   markComplete(lessonId)        -> Promise<void>   mark a lesson done
 *   isComplete(lessonId)          -> Promise<boolean> has it been marked done
 *   exportProgress()              -> Promise<string>  JSON progress token
 *   importProgress(json)          -> Promise<void>    restore from a token
 *
 * Plus UI/runtime helpers used by the lesson pages:
 *   downloadProgressToken()       trigger a browser download of the token
 *   renderProgressNote(el)        inject the "not authoritative" banner
 *   installPyodideLoadWatchdog()  show a retry note if the kernel never loads
 *   installGradingWatchdog()      restart the kernel worker on a runaway cell
 *
 * No framework, no build step — vanilla ES module-free script. Load with a
 * plain <script src="_static/gating.js"></script>; everything hangs off the
 * global `CourseGating` object (and a few convenience globals for notebooks).
 */
(function (global) {
  "use strict";

  // ---------------------------------------------------------------------------
  // Constants / configuration
  // ---------------------------------------------------------------------------
  var DB_NAME = "slm-course-progress";
  var DB_VERSION = 1;
  var STORE = "completion"; // object store keyed by lessonId
  var LS_KEY = "slm-course-progress"; // localStorage fallback key
  var TOKEN_VERSION = 1; // bump if the token schema ever changes

  // Canonical lesson ids — these match the shipped notebook filenames so a
  // lesson page can call markComplete() with a stable, human-readable id.
  var LESSON_IDS = [
    "s1_tokenizer_warmup",
    "m00_orientation",
    "m01_numpy",
    "m02_neuron",
    "m03_autograd",
    "m04_bigram",
    "m05_mlp_lm",
    "m06_attention",
    "m07_gpt",
    "m08_train",
    "m09_capstone",
    "m10_whats_next",
  ];

  // Pyodide is expected to cold-load within this window on a first visit.
  // It is a multi-MB download, so we are generous; the point is to never let
  // the page hang silently forever.
  var PYODIDE_LOAD_TIMEOUT_MS = 90 * 1000;

  // A single student grading cell that runs longer than this is treated as a
  // runaway loop. Pyodide is single-Web-Worker: a CPU-bound Python loop cannot
  // be cooperatively interrupted, so the only recovery is to terminate and
  // restart the worker (documented below in installGradingWatchdog).
  var GRADING_TIMEOUT_MS = 30 * 1000;

  // ---------------------------------------------------------------------------
  // Storage layer: IndexedDB primary, localStorage fallback.
  //
  // We treat IndexedDB as the primary store because it survives larger payloads
  // and is the JupyterLite default, but every code path degrades to
  // localStorage if IndexedDB is unavailable (private mode, old browser, or an
  // open error). Neither is authoritative — see the file header.
  // ---------------------------------------------------------------------------

  var _idbAvailable = typeof global.indexedDB !== "undefined";

  function openDB() {
    // Resolves with an IDBDatabase, or rejects so callers fall back to LS.
    return new Promise(function (resolve, reject) {
      if (!_idbAvailable) {
        reject(new Error("IndexedDB unavailable"));
        return;
      }
      var req = global.indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = function () {
        var db = req.result;
        if (!db.objectStoreNames.contains(STORE)) {
          db.createObjectStore(STORE); // keyed externally by lessonId
        }
      };
      req.onsuccess = function () {
        resolve(req.result);
      };
      req.onerror = function () {
        reject(req.error || new Error("IndexedDB open failed"));
      };
    });
  }

  function idbPut(lessonId, record) {
    return openDB().then(function (db) {
      return new Promise(function (resolve, reject) {
        var tx = db.transaction(STORE, "readwrite");
        tx.objectStore(STORE).put(record, lessonId);
        tx.oncomplete = function () {
          db.close();
          resolve();
        };
        tx.onerror = function () {
          db.close();
          reject(tx.error || new Error("IndexedDB write failed"));
        };
      });
    });
  }

  function idbGetAll() {
    // Returns a plain object { lessonId: record, ... }.
    return openDB().then(function (db) {
      return new Promise(function (resolve, reject) {
        var out = {};
        var tx = db.transaction(STORE, "readonly");
        var store = tx.objectStore(STORE);
        // openCursor is the most broadly supported way to read keys+values.
        var req = store.openCursor();
        req.onsuccess = function () {
          var cursor = req.result;
          if (cursor) {
            out[cursor.key] = cursor.value;
            cursor.continue();
          } else {
            db.close();
            resolve(out);
          }
        };
        req.onerror = function () {
          db.close();
          reject(req.error || new Error("IndexedDB read failed"));
        };
      });
    });
  }

  // ---- localStorage fallback helpers (mirror the IDB record shape) ----------
  function lsReadAll() {
    try {
      var raw = global.localStorage.getItem(LS_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (e) {
      return {};
    }
  }

  function lsWriteAll(obj) {
    try {
      global.localStorage.setItem(LS_KEY, JSON.stringify(obj));
      return true;
    } catch (e) {
      return false; // quota / disabled storage — non-fatal, just no persistence
    }
  }

  // Unified read of the whole progress map, preferring IndexedDB and falling
  // back to localStorage. Always resolves (never rejects) with an object.
  function readAll() {
    return idbGetAll().catch(function () {
      return lsReadAll();
    });
  }

  // Unified write of a single lesson record to whichever store is available.
  function writeRecord(lessonId, record) {
    return idbPut(lessonId, record).catch(function () {
      var all = lsReadAll();
      all[lessonId] = record;
      lsWriteAll(all);
    });
  }

  // ---------------------------------------------------------------------------
  // navigator.storage.persist() — request durable storage so the browser is
  // less likely to evict our IndexedDB under storage pressure. This is a hint,
  // not a guarantee, which is exactly why the JSON token is the real backup.
  // ---------------------------------------------------------------------------
  function requestPersistentStorage() {
    if (
      global.navigator &&
      global.navigator.storage &&
      typeof global.navigator.storage.persist === "function"
    ) {
      return global.navigator.storage.persist().then(
        function (granted) {
          return !!granted;
        },
        function () {
          return false;
        }
      );
    }
    return Promise.resolve(false);
  }

  // ---------------------------------------------------------------------------
  // Public API: completion flags
  // ---------------------------------------------------------------------------
  function markComplete(lessonId) {
    if (!lessonId) return Promise.reject(new Error("lessonId required"));
    var record = { complete: true, ts: new Date().toISOString() };
    // Opportunistically ask for persistent storage the first time we write.
    requestPersistentStorage();
    return writeRecord(lessonId, record);
  }

  function isComplete(lessonId) {
    if (!lessonId) return Promise.resolve(false);
    return readAll().then(function (all) {
      var rec = all[lessonId];
      return !!(rec && rec.complete);
    });
  }

  // ---------------------------------------------------------------------------
  // Public API: progress token (the durable, user-held backup)
  //
  // The token is a small JSON document the learner downloads (or copies) and
  // can paste back later — independent of any browser storage. This is the
  // authoritative-to-the-learner artifact; IndexedDB is just a cache of it.
  // ---------------------------------------------------------------------------
  function exportProgress() {
    return readAll().then(function (all) {
      var token = {
        v: TOKEN_VERSION,
        course: "build-an-slm-from-scratch",
        exported: new Date().toISOString(),
        lessons: all, // { lessonId: { complete, ts }, ... }
      };
      return JSON.stringify(token, null, 2);
    });
  }

  function importProgress(json) {
    // Accept either a JSON string or an already-parsed object. Validate
    // defensively — a pasted token may be truncated or hand-edited.
    return new Promise(function (resolve, reject) {
      var token;
      try {
        token = typeof json === "string" ? JSON.parse(json) : json;
      } catch (e) {
        reject(new Error("Progress token is not valid JSON"));
        return;
      }
      if (!token || typeof token !== "object" || !token.lessons) {
        reject(new Error("Progress token is missing a 'lessons' field"));
        return;
      }
      // Merge imported completion flags into local storage. We only ever set
      // flags true (a token restore should not un-complete a lesson the
      // learner finished in this browser).
      var writes = [];
      Object.keys(token.lessons).forEach(function (lessonId) {
        var rec = token.lessons[lessonId];
        if (rec && rec.complete) {
          writes.push(
            writeRecord(lessonId, {
              complete: true,
              ts: rec.ts || new Date().toISOString(),
            })
          );
        }
      });
      Promise.all(writes).then(function () {
        resolve();
      }, reject);
    });
  }

  // Convenience: trigger a real file download of the current progress token.
  function downloadProgressToken(filename) {
    return exportProgress().then(function (jsonStr) {
      var blob = new global.Blob([jsonStr], { type: "application/json" });
      var url = global.URL.createObjectURL(blob);
      var a = global.document.createElement("a");
      a.href = url;
      a.download = filename || "slm-course-progress.json";
      global.document.body.appendChild(a);
      a.click();
      global.document.body.removeChild(a);
      // Revoke on the next tick so the click has a chance to start.
      global.setTimeout(function () {
        global.URL.revokeObjectURL(url);
      }, 0);
    });
  }

  // ---------------------------------------------------------------------------
  // The "not authoritative" banner. Lessons call renderProgressNote(el) to
  // inject a plain-language note explaining the persistence model.
  // ---------------------------------------------------------------------------
  function renderProgressNote(targetEl) {
    var el =
      targetEl ||
      (global.document && global.document.getElementById("progress-note"));
    if (!el) return;
    el.innerHTML =
      '<div class="slm-progress-note" role="note" ' +
      'style="border:1px solid #d0d7de;border-left:4px solid #fb8500;' +
      "background:#fff8f0;padding:0.75rem 1rem;border-radius:6px;" +
      'font-size:0.9rem;line-height:1.45;">' +
      "<strong>Your progress is saved in this browser only.</strong> " +
      "Completion checkmarks live on this device and can be lost if you " +
      "clear your browser data or switch devices. They are a convenience, " +
      "not an official record of completion. " +
      "<strong>To keep your progress, use “Download my progress”</strong> " +
      "and save the JSON file — you can paste it back any time to restore." +
      "</div>";
  }

  // ---------------------------------------------------------------------------
  // Pyodide load-failure fallback.
  //
  // The in-browser kernel is a multi-MB download. On a slow connection, an old
  // browser, or a flaky CDN it may never finish. Rather than let lesson 0 hang
  // forever on a beginner, we arm a watchdog: if the kernel has not signalled
  // "ready" within PYODIDE_LOAD_TIMEOUT_MS, we render a plain-language retry +
  // a minimum-browser note.
  //
  // The page is expected to call CourseGating.markKernelReady() once the
  // JupyterLite/Pyodide kernel reports ready (e.g. from a kernel-status hook).
  // ---------------------------------------------------------------------------
  var _kernelReady = false;
  var _pyodideTimer = null;

  function markKernelReady() {
    _kernelReady = true;
    if (_pyodideTimer) {
      global.clearTimeout(_pyodideTimer);
      _pyodideTimer = null;
    }
  }

  function showKernelLoadFailure(mountEl) {
    var el =
      mountEl ||
      (global.document &&
        global.document.getElementById("kernel-fallback"));
    if (!el) return;
    el.innerHTML =
      '<div class="slm-kernel-fallback" role="alert" ' +
      'style="border:1px solid #d0d7de;border-left:4px solid #d00000;' +
      "background:#fff5f5;padding:1rem;border-radius:6px;font-size:0.95rem;" +
      'line-height:1.5;">' +
      "<strong>The in-browser Python kernel did not finish loading.</strong>" +
      "<p style=\"margin:0.5rem 0;\">This first-time download is a few " +
      "megabytes and can stall on a slow or filtered connection.</p>" +
      "<ul style=\"margin:0.5rem 0 0.75rem 1.25rem;\">" +
      "<li>Use a recent desktop browser — Chrome, Edge, Firefox, or " +
      "Safari from the last couple of years (WebAssembly is required).</li>" +
      "<li>Check that a network filter or extension is not blocking the " +
      "kernel download.</li>" +
      "</ul>" +
      '<button type="button" id="slm-kernel-retry" ' +
      'style="cursor:pointer;padding:0.4rem 0.9rem;border-radius:6px;' +
      'border:1px solid #888;background:#f6f8fa;">Retry loading the kernel</button>' +
      "</div>";
    var btn = el.querySelector("#slm-kernel-retry");
    if (btn) {
      btn.addEventListener("click", function () {
        global.location.reload();
      });
    }
  }

  function installPyodideLoadWatchdog(mountEl, timeoutMs) {
    _kernelReady = false;
    if (_pyodideTimer) global.clearTimeout(_pyodideTimer);
    _pyodideTimer = global.setTimeout(function () {
      if (!_kernelReady) showKernelLoadFailure(mountEl);
    }, timeoutMs || PYODIDE_LOAD_TIMEOUT_MS);
    return _pyodideTimer;
  }

  // ---------------------------------------------------------------------------
  // Grading watchdog (runaway student cell).
  //
  // DOCUMENTED LIMITATION: Pyodide runs Python on a single Web Worker. A
  // CPU-bound Python loop (e.g. `while True: pass`) holds that worker's thread
  // and CANNOT be interrupted cooperatively from the main thread — there is no
  // signal Python will check mid-loop. The only reliable recovery is to
  // terminate the Worker and start a fresh one. That loses in-memory kernel
  // state (the learner must re-run their cells), which is an acceptable cost
  // for not freezing the tab.
  //
  // This helper wraps the act of running a grading cell: start a timer; if the
  // cell does not report completion within GRADING_TIMEOUT_MS, invoke the
  // caller-supplied restartWorker() and surface a clear FAIL message. The page
  // supplies how to actually run the cell and how to restart its worker, since
  // those are JupyterLite-internal; this module owns only the timing + UX.
  // ---------------------------------------------------------------------------
  function installGradingWatchdog(opts) {
    opts = opts || {};
    var timeoutMs = opts.timeoutMs || GRADING_TIMEOUT_MS;
    var restartWorker =
      typeof opts.restartWorker === "function" ? opts.restartWorker : null;
    var onTimeout =
      typeof opts.onTimeout === "function" ? opts.onTimeout : null;

    // runCell(runner) — `runner` returns a Promise that resolves when the
    // grading cell finishes. Returns a Promise that either resolves with the
    // cell's result, or rejects with a timeout error after restarting the
    // worker so the kernel is usable again.
    function runCell(runner) {
      var settled = false;
      return new Promise(function (resolve, reject) {
        var timer = global.setTimeout(function () {
          if (settled) return;
          settled = true;
          // Kill and replace the frozen worker so the page stays usable.
          if (restartWorker) {
            try {
              restartWorker();
            } catch (e) {
              /* swallow — we are already in the failure path */
            }
          }
          var msg =
            "Your code ran longer than " +
            Math.round(timeoutMs / 1000) +
            "s and was stopped. The in-browser kernel was restarted, so " +
            "you'll need to re-run your cells. A common cause is an " +
            "accidental infinite loop — check your loop conditions.";
          if (onTimeout) onTimeout(msg);
          reject(new Error(msg));
        }, timeoutMs);

        Promise.resolve()
          .then(runner)
          .then(
            function (result) {
              if (settled) return;
              settled = true;
              global.clearTimeout(timer);
              resolve(result);
            },
            function (err) {
              if (settled) return;
              settled = true;
              global.clearTimeout(timer);
              reject(err);
            }
          );
      });
    }

    return { runCell: runCell };
  }

  // ---------------------------------------------------------------------------
  // Export the public surface.
  // ---------------------------------------------------------------------------
  var CourseGating = {
    LESSON_IDS: LESSON_IDS,
    // core API
    markComplete: markComplete,
    isComplete: isComplete,
    exportProgress: exportProgress,
    importProgress: importProgress,
    // helpers
    downloadProgressToken: downloadProgressToken,
    renderProgressNote: renderProgressNote,
    requestPersistentStorage: requestPersistentStorage,
    markKernelReady: markKernelReady,
    installPyodideLoadWatchdog: installPyodideLoadWatchdog,
    showKernelLoadFailure: showKernelLoadFailure,
    installGradingWatchdog: installGradingWatchdog,
  };

  global.CourseGating = CourseGating;

  // Also expose the four core functions as bare globals so notebook cells can
  // call markComplete()/isComplete()/exportProgress()/importProgress() directly.
  global.markComplete = markComplete;
  global.isComplete = isComplete;
  global.exportProgress = exportProgress;
  global.importProgress = importProgress;
})(typeof window !== "undefined" ? window : this);
