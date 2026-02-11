import com.vanniktech.maven.publish.SonatypeHost

plugins {
    alias(libs.plugins.android.library)
    id("com.vanniktech.maven.publish") version "0.28.0"
}

android {
    namespace = "ai.moonshine.voice"
    compileSdk = 35
    ndkVersion = "25.2.9519653"

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

mavenPublishing {
    publishToMavenCentral(SonatypeHost.CENTRAL_PORTAL)
    signAllPublications()
    
    coordinates("ai.moonshine", "moonshine-voice", "0.0.48")

    pom {
        name.set("Moonshine Voice")
        description.set("Build fast, local and accurate transcription and voice interfaces with Moonshine Voice.")
        url.set("https://github.com/moonshine-ai/moonshine")
        licenses {
            license {
                name.set("MIT")
                url.set("https://opensource.org/licenses/MIT")
            }
        }
        developers {
            developer {
                id.set("moonshine-ai")
                name.set("Moonshine AI")
                email.set("contact@moonshine.ai")
            }
        }
       scm {
            url.set("https://github.com/moonshine-ai/moonshine")
            connection.set("scm:git:git://github.com/moonshine-ai/moonshine.git")
            developerConnection.set("scm:git:ssh://git@github.com/moonshine-ai/moonshine.git")
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