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

  async function ensureInit() {
    if (initialised) return { app, auth, fbMod };
    if (pendingInit) return pendingInit;
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
    return ensureInit();
  }

  function currentUser() {
    if (!auth || !auth.currentUser) return null;
    return userToSession(auth.currentUser);
  }

  function onAuthChange(callback) {
    // Must be called after init() so the auth instance exists.
    if (!auth) {
      try { callback(null); } catch (e) { /* ignore */ }
      return () => {};
    }
    return fbMod.onAuthStateChanged(auth, (u) => {
      try { callback(u ? userToSession(u) : null); } catch (e) { console.error(e); }
    });
  }

  async function signIn(email, password) {
    await ensureInit();
    const cred = await fbMod.signInWithEmailAndPassword(auth, email, password);
    return userToSession(cred.user);
  }

  async function signUp(name, email, password) {
    await ensureInit();
    const cred = await fbMod.createUserWithEmailAndPassword(auth, email, password);
    if (name && cred.user) {
      try { await fbMod.updateProfile(cred.user, { displayName: name }); } catch (e) { /* non-fatal */ }
    }
    return userToSession(cred.user);
  }

  async function signInWithGoogle() {
    await ensureInit();
    const cred = await fbMod.signInWithPopup(auth, googleProvider);
    return userToSession(cred.user);
  }

  async function resetPassword(email) {
    await ensureInit();
    await fbMod.sendPasswordResetEmail(auth, email);
  }

  async function signOut() {
    await ensureInit();
    await fbMod.signOut(auth);
  }

  window.AuthAPI = {
    init, signIn, signUp, signInWithGoogle, resetPassword, signOut,
    onAuthChange, currentUser, friendlyError
  };
})();
