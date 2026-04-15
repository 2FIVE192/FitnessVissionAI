import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

// Read backend URLs from local.properties so they never have to be hardcoded in source.
// Defaults point to the Android emulator's loopback alias (10.0.2.2 = host machine).
val localProps = Properties()
val localPropsFile = rootProject.file("local.properties")
if (localPropsFile.exists()) localProps.load(localPropsFile.inputStream())
val backendUrl    = localProps.getProperty("backend.url",    "http://10.0.2.2:3000")
val mlBackendUrl  = localProps.getProperty("ml.backend.url", "http://10.0.2.2:8000")

android {
    namespace = "com.app.fitness.mobile"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.app.fitness.mobile"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"

        buildConfigField("String", "BACKEND_URL",    "\"$backendUrl\"")
        buildConfigField("String", "ML_BACKEND_URL", "\"$mlBackendUrl\"")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }
}

dependencies {
    // fitness sdk (resolved via includeBuild in settings)
    implementation("com.app.fitness:fitness-sdk:1.0.0")

    // android core
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime)
    implementation(libs.androidx.lifecycle.viewmodel)
    implementation(libs.androidx.activity.compose)

    // compose
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.icons.extended)
    implementation(libs.navigation.compose)

    // coroutines
    implementation(libs.coroutines.android)
}
