plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "ai.moonshine.androidtranscriber"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.moonshine.androidtranscriber"
        minSdk = 35
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    implementation(libs.moonshine.voice)
    implementation(libs.appcompat)
    implementation(libs.material)
}
