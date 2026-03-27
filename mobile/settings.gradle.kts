val localProps = file("local.properties")
if (!localProps.exists()) {
    val androidHome = System.getenv("ANDROID_HOME")
        ?: System.getenv("ANDROID_SDK_ROOT")
        ?: "${System.getProperty("user.home")}\\AppData\\Local\\Android\\Sdk"
    val sdkDir = File(androidHome)
    if (sdkDir.exists()) {
        localProps.writeText("sdk.dir=${androidHome.replace("\\", "\\\\")}\n")
    }
}

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "FitnessApp"
include(":app")

// pull in the sdk from the sibling directory so we can use it as a local dep
includeBuild("../fitness-sdk") {
    dependencySubstitution {
        substitute(module("com.app.fitness:fitness-sdk")).using(project(":sdk"))
    }
}
