plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "ai.moonshine.voice"
    compileSdk = 35

    defaultConfig {
        minSdk = 35

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
        ndk {
            // Only build for ARM64 to match the app module
            abiFilters += listOf("arm64-v8a")
        }
    }

    sourceSets {
        getByName("main") {
            java.srcDirs("android/java/main")
        }
        getByName("androidTest") {
            java.srcDirs("android/java/androidTest")
            assets.srcDirs("test-assets")
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    externalNativeBuild {
        cmake {
            path = file("core/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}