plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    `maven-publish`
}

android {
    namespace = "com.app.fitness"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
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

    kotlinOptions {
        jvmTarget = "11"
    }

    // Expose the full sources jar so consumers see doc comments
    publishing {
        singleVariant("release") {
            withSourcesJar()
            withJavadocJar()
        }
    }
}

dependencies {
    api(libs.androidx.core.ktx)
    api(libs.coroutines.core)
    api(libs.coroutines.android)
    api(libs.work.runtime.ktx)

    api(libs.retrofit)
    api(libs.retrofit.converter.gson)
    api(libs.okhttp.logging)
    api(libs.gson)
}

// ── Maven publish ─────────────────────────────────────────────────────────────
// Published coordinates: com.app.fitness:fitness-sdk:1.0.0
//
// Local Maven (~/.m2):      ./gradlew publishToMavenLocal
// GitHub Packages:          set env vars GITHUB_ACTOR / GITHUB_TOKEN and run
//                           ./gradlew publish
//
// JitPack:  tag & push a release — JitPack builds automatically.
//           Add to consumer app:
//             repositories { maven("https://jitpack.io") }
//             implementation("com.github.YourOrg:fitness-sdk:1.0.0")
// ─────────────────────────────────────────────────────────────────────────────

afterEvaluate {
    publishing {
        publications {
            create<MavenPublication>("release") {
                from(components["release"])
                groupId    = "com.app.fitness"
                artifactId = "fitness-sdk"
                version    = "1.0.0"

                pom {
                    name.set("Fitness SDK")
                    description.set("Kotlin Android library providing interfaces and HTTP implementations for the Fitness backend API.")
                    licenses {
                        license {
                            name.set("MIT License")
                            url.set("https://opensource.org/licenses/MIT")
                        }
                    }
                }
            }
        }

        repositories {
            // Local Maven cache — always available
            mavenLocal()

            // GitHub Packages — set GITHUB_ACTOR and GITHUB_TOKEN env vars
            maven {
                name = "GitHubPackages"
                url  = uri("https://maven.pkg.github.com/YourOrg/alexander-mobile-backend")
                credentials {
                    username = System.getenv("GITHUB_ACTOR") ?: ""
                    password = System.getenv("GITHUB_TOKEN") ?: ""
                }
            }
        }
    }
}
