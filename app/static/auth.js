// Firebase Authentication integration for the DR Screening app.
// Uses the modular Web SDK v10+ via the CDN.
// Exposes window.AuthAPI with: init(), signIn, signUp, signInWithGoogle,
// resetPassword, signOut, onAuthChange, currentUser.

(function () {
  "use strict";

  let app = null;
  let auth = null;
  let fbMod = null;
  let googleProvider = null;
  let initialised = false;
  let pendingInit = null;

  // Friendly error messages for the most common Firebase Auth error codes.
  const ERROR_MESSAGES = {
    "auth/invalid-email":        "Please enter a valid email address.",
    "auth/user-not-found":       "No account found with this email.",
    "auth/wrong-password":       "Incorrect password. Please try again.",
    "auth/invalid-credential":   "Incorrect email or password. Please try again.",
    "auth/invalid-login-credentials": "Incorrect email or password. Please try again.",
    "auth/user-disabled":        "This account has been disabled. Contact your administrator.",
    "auth/email-already-in-use": "An account with this email already exists. Try logging in instead.",
    "auth/weak-password":        "Password is too weak. Use at least 6 characters.",
    "auth/missing-password":     "Please enter a password.",
    "auth/too-many-requests":    "Too many attempts. Please wait a moment and try again.",
    "auth/network-request-failed": "Network error. Please check your connection and retry.",
    "auth/popup-closed-by-user": "Sign-in popup was closed before completion.",
    "auth/popup-blocked":        "Sign-in popup was blocked by the browser. Please allow popups for this site.",
    "auth/cancelled-popup-request": "Sign-in was cancelled.",
    "auth/operation-not-allowed":"This sign-in method is not enabled. Contact the administrator.",
    "auth/requires-recent-login":"Please sign in again to perform this action."
  };

  function friendlyError(err) {
    if (!err) return "Something went wrong. Please try again.";
    const code = err.code || "";
    if (ERROR_MESSAGES[code]) return ERROR_MESSAGES[code];
    return err.message || "Something went wrong. Please try again.";
  }

  function userToSession(user) {
    if (!user) return null;
    return {
      uid: user.uid,
      email: user.email || "",
      name: user.displayName || (user.email ? user.email.split("@")[0] : "Clinician"),
      photoURL: user.photoURL || "",
      emailVerified: !!user.emailVerified,
      provider: (user.providerData && user.providerData[0] && user.providerData[0].providerId) || "firebase"
    };
  }

  let authStateCallback = null;

  function triggerAuthChange(user) {
    if (authStateCallback) {
      try {
        authStateCallback(user);
      } catch (e) {
        console.error("RETINACARE_TRACE STEP_10 triggerAuthChange error");
      }
    }
  }

  async function ensureInit() {
    console.log("RETINACARE_AUTH_DEBUG: ensureInit called");
    if (initialised) return { app, auth, fbMod };
    if (pendingInit) return pendingInit;

    // When running inside Android WebView with native bridge
    if (window.AndroidBridge) {
      console.log("RETINACARE_AUTH_DEBUG: Detected AndroidBridge");
      initialised = true;
      return { app: null, auth: null, fbMod: null };
    }

    pendingInit = (async () => {
      const cfg = window.FIREBASE_CONFIG;
      if (!cfg || !cfg.apiKey) {
        throw new Error("Firebase config missing. Did /static/firebase-config.js load?");
      }
      const fbAppMod  = await import("https://www.gstatic.com/firebasejs/10.13.2/firebase-app.js");
      const fbAuthMod = await import("https://www.gstatic.com/firebasejs/10.13.2/firebase-auth.js");
      app  = fbAppMod.initializeApp(cfg);
      auth = fbAuthMod.getAuth(app);
      googleProvider = new fbAuthMod.GoogleAuthProvider();
      fbMod = fbAuthMod;
      initialised = true;
      return { app, auth, fbMod };
    })();
    return pendingInit;
  }

  async function init() {
    const initResult = await ensureInit();
    if (window.AndroidBridge) {
      // For Android native mode, trigger auth listener with any existing session
      const s = sessionStorage.getItem("sessionUser");
      if (s) {
        try {
          const userParsed = JSON.parse(s);
          triggerAuthChange(userParsed);
        } catch (e) { console.error("RETINACARE_AUTH_DEBUG: init session parse error", e); }
      }
    }
    return initResult;
  }

  // Demo-only native login for Android APK (Phase 6B)
  async function nativeLogin(email, password) {
    console.log("RETINACARE_TRACE STEP_4 nativeLogin called");
    if (!window.AndroidBridge || !window.AndroidBridge.login) {
      console.error("RETINACARE_AUTH_DEBUG: AndroidBridge or login missing");
      throw new Error("Native Android bridge not available");
    }
    console.log("RETINACARE_TRACE STEP_5 AndroidBridge.login called");
    const raw = window.AndroidBridge.login(email, password);
    console.log("RETINACARE_TRACE STEP_6 AndroidBridge.login returned");
    console.log("RETINACARE_TRACE STEP_5 nativeLogin raw response (type=" + typeof raw + ")");
    // Robust parsing: login() returns JSON string; handle already-parsed object or null
    let result;
    if (raw === null || raw === undefined) {
      result = null;
    } else if (typeof raw === "object") {
      result = raw;
    } else {
      try {
        result = JSON.parse(raw);
      } catch (e) {
        console.error("RETINACARE_AUTH_DEBUG: JSON.parse failed on raw=", raw, "error=", e);
        throw new Error("Invalid response from native login: " + String(raw));
      }
    }
    console.log("RETINACARE_TRACE STEP_8 nativeLogin parsed success=" + (result && result.success) + " error=" + (result ? result.error : "none"));

    if (result && result.success) {
      console.log("RETINACARE_TRACE STEP_8 nativeLogin success true");
      return {
        uid: "demo-user",
        email: result.email || email,
        name: result.name || "Dr. Demo",
        photoURL: "",
        emailVerified: true,
        provider: "native-demo",
        role: result.role || "clinician"
      };
    }
    console.error("RETINACARE_AUTH_DEBUG: nativeLogin failed", result?.error);
    throw new Error(result ? (result.error || "Login failed") : "Login failed");
  }

  function currentUser() {
    if (window.AndroidBridge) {
      // Native session managed outside Firebase; return from sessionStorage
      try {
        const s = sessionStorage.getItem("sessionUser");
        if (s) return JSON.parse(s);
      } catch (e) { /* ignore */ }
      return null;
    }
    if (!auth || !auth.currentUser) return null;
    return userToSession(auth.currentUser);
  }

  function onAuthChange(callback) {
    console.log("RETINACARE_TRACE STEP_11 onAuthChange registered");
    console.log("RETINACARE_AUTH_DEBUG: onAuthChange called");
    if (window.AndroidBridge) {
      // For native demo, trigger once with current session user
      try {
        const s = sessionStorage.getItem("sessionUser");
        console.log("RETINACARE_AUTH_DEBUG: onAuthChange native, sessionUser:", s);
        callback(s ? JSON.parse(s) : null);
      } catch (e) { console.error("RETINACARE_AUTH_DEBUG: onAuthChange error", e); callback(null); }
      // Return a no-op unsubscribe for API consistency
      return () => {};
    }
    if (!auth) {
      try { callback(null); } catch (e) { /* ignore */ }
      return () => {};
    }
    return fbMod.onAuthStateChanged(auth, (u) => {
      try { callback(u ? userToSession(u) : null); } catch (e) { console.error(e); }
    });
  }

  async function signIn(email, password) {
    console.log("RETINACARE_TRACE STEP_2 signIn called");
    if (window.AndroidBridge) {
      const user = await nativeLogin(email, password);
      console.log("RETINACARE_TRACE STEP_3 sessionStorage set");
      sessionStorage.setItem("sessionUser", JSON.stringify(user));

      console.log("RETINACARE_TRACE STEP_10 triggerAuthChange");
      triggerAuthChange(user);
      return user;
    }
    await ensureInit();
    const cred = await fbMod.signInWithEmailAndPassword(auth, email, password);
    return userToSession(cred.user);
  }

  async function signUp(name, email, password) {
    if (window.AndroidBridge) {
      throw new Error("Sign-up is not supported in demo mode. Use the demo account: demo@retinacare.ai / Demo@1234");
    }
    await ensureInit();
    const cred = await fbMod.createUserWithEmailAndPassword(auth, email, password);
    if (name && cred.user) {
      try { await fbMod.updateProfile(cred.user, { displayName: name }); } catch (e) { /* non-fatal */ }
    }
    return userToSession(cred.user);
  }

  async function signInWithGoogle() {
    if (window.AndroidBridge) {
      throw new Error("Google sign-in not available in demo mode.");
    }
    await ensureInit();
    const cred = await fbMod.signInWithPopup(auth, googleProvider);
    return userToSession(cred.user);
  }

  async function resetPassword(email) {
    if (window.AndroidBridge) {
      throw new Error("Password reset not available in demo mode.");
    }
    await ensureInit();
    await fbMod.sendPasswordResetEmail(auth, email);
  }

  async function signOut() {
    if (window.AndroidBridge) {
      sessionStorage.removeItem("sessionUser");
      return;
    }
    await ensureInit();
    await fbMod.signOut(auth);
  }

  window.AuthAPI = {
    init, signIn, signUp, signInWithGoogle, resetPassword, signOut,
    onAuthChange, currentUser, friendlyError, nativeLogin
  };
})();
