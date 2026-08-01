plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "ai.moonshine.examples.agentflow"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.moonshine.examples.agentflow"
        minSdk = 26
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
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation(libs.moonshine.voice)
    implementation(libs.appcompat)
    implementation(libs.material)
}
