# Android Project README
**SIH 2026 - DR Screening Android Application**

## Project Setup

This is a native Android application built with:
- **Language:** Kotlin
- **UI Framework:** Jetpack Compose (Material 3)
- **Architecture:** Single-Activity MVVM
- **Build System:** Gradle (Kotlin DSL)

## Configuration

- **Package:** `com.sih2026.drscreening`
- **Min SDK:** API 26 (Android 8.0 Oreo)
- **Target SDK:** API 34
- **Compile SDK:** API 34

## Key Dependencies

### Core
- Jetpack Compose with Material 3
- Kotlin Coroutines
- Lifecycle & ViewModel

### Features
- **CameraX:** Retinal image capture
- **Room:** Offline local database
- **TensorFlow Lite:** On-device AI inference
- **Navigation Compose:** Screen navigation

### Permissions
- `CAMERA` - Required for retinal image capture
- **NO INTERNET PERMISSION** - Fully offline application

## Localization
- English (default)
- Kannada (ಕನ್ನಡ) - `values-kn/`

## Build Instructions

### Via Android Studio
1. Open Android Studio
2. Open project: `File → Open → select android/ folder`
3. Wait for Gradle sync
4. Run: `Run → Run 'app'`

### Via Command Line
```bash
cd android
./gradlew assembleDebug  # Windows: gradlew.bat assembleDebug
```

## Project Structure
```
android/
├── app/
│   ├── src/main/
│   │   ├── java/com/sih2026/drscreening/
│   │   │   ├── MainActivity.kt
│   │   │   └── ui/theme/
│   │   ├── res/
│   │   │   ├── values/
│   │   │   └── values-kn/  (Kannada localization)
│   │   └── AndroidManifest.xml
│   ├── build.gradle.kts
│   └── proguard-rules.pro
├── gradle/
│   ├── libs.versions.toml
│   └── wrapper/
├── build.gradle.kts
└── settings.gradle.kts
```

## Current Status
**Phase 1: Project Foundation** ✅

- [x] Android project structure created
- [x] Jetpack Compose with Material 3 setup
- [x] Kotlin configuration
- [x] TensorFlow Lite dependency added
- [x] CameraX dependency added
- [x] Room database dependency added
- [x] English + Kannada localization structure
- [x] No internet permission (offline-only)

## Next Steps (Phase 2)
- Implement MVVM architecture (ViewModels, Repositories)
- Implement Room database schema (ScreeningResult entity)
- Integrate TensorFlow Lite runtime
- Implement CameraX preview and capture
- Design screen navigation flow

---

**Target Deployment:** Rural and Semi-Urban Karnataka PHCs
