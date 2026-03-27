package com.app.fitness.mobile.navigation

import android.Manifest
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.FitnessCenter
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Person
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.*
import com.app.fitness.FitnessSdk
import com.app.fitness.mobile.screen.*
import com.app.fitness.mobile.viewmodel.*

// route names
object Routes {

    const val AUTH = "auth"
    const val LOGIN = "login"

    const val REGISTER = "register"
    /*const val REGISTER_GRAPH = "register_graph"
    const val REGISTER_START = "register_start"
    const val REGISTER_BODY = "register_body"
    const val REGISTER_GOALS = "register_goals"*/
    const val HOME = "home"
    const val WORKOUTS = "workouts"
    const val WORKOUT_DETAIL = "workouts/{workoutId}"
    const val PROFILE = "profile"
    const val FITVISION = "fitvision"
}

// bottom nav items
data class BottomNavItem(val route: String, val label: String, val icon: @Composable () -> Unit)

@Composable
fun AppNavGraph(sdk: FitnessSdk) {
    val navController = rememberNavController()

    // check if user has a stored token to decide start destination
    val startDest = if (sdk.hasStoredToken) Routes.HOME else Routes.AUTH

    // ACTIVITY_RECOGNITION (API 29+) and POST_NOTIFICATIONS (API 33+) are both
    // runtime permissions required before starting the step counter service.
    val stepPermissionsLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        // Only start if ACTIVITY_RECOGNITION was granted (sensor will silently fail otherwise)
        val activityGranted = results[Manifest.permission.ACTIVITY_RECOGNITION] != false
        if (activityGranted) sdk.startStepCounting()
    }

    fun startStepCountingWithPermission() {
        val permissions = buildList {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                add(Manifest.permission.ACTIVITY_RECOGNITION)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU)
                add(Manifest.permission.POST_NOTIFICATIONS)
        }
        if (permissions.isEmpty()) {
            sdk.startStepCounting()
        } else {
            stepPermissionsLauncher.launch(permissions.toTypedArray())
        }
    }

    val bottomItems = listOf(
        BottomNavItem(Routes.HOME, "Главная") { Icon(Icons.Default.Home, contentDescription = "Главная") },
        BottomNavItem(Routes.WORKOUTS, "Тренировки") { Icon(Icons.Default.FitnessCenter, contentDescription = "Тренировки") },
        BottomNavItem(Routes.FITVISION, "FitVision") { Icon(Icons.Default.AutoAwesome, contentDescription = "FitVision") },
        BottomNavItem(Routes.PROFILE, "Профиль") { Icon(Icons.Default.Person, contentDescription = "Профиль") },
    )

    // track whether we're on a screen that should show bottom nav
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route
    val showBottomBar = currentRoute in listOf(Routes.HOME, Routes.WORKOUTS, Routes.FITVISION, Routes.PROFILE)

    Scaffold(
        bottomBar = {
            if (showBottomBar) {
                NavigationBar {
                    bottomItems.forEach { item ->
                        val selected = navBackStackEntry?.destination?.hierarchy?.any {
                            it.route == item.route
                        } == true

                        NavigationBarItem(
                            icon = item.icon,
                            label = { Text(item.label) },
                            selected = selected,
                            onClick = {
                                navController.navigate(item.route) {
                                    popUpTo(navController.graph.findStartDestination().id) {
                                        saveState = true
                                    }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            }
                        )
                    }
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = startDest,
            modifier = Modifier.padding(innerPadding)
        ) {
            composable(Routes.AUTH) {

                val vm: LoginViewModel = viewModel(
                    factory = LoginViewModelFactory(sdk.auth)
                )

                AuthStartScreen(
                    viewModel = vm,
                    onLoginClick = {
                        navController.navigate(Routes.LOGIN)
                    },
                    onRegisterClick = {
                        navController.navigate(Routes.REGISTER)
                    }
                )
            }


            composable(Routes.LOGIN) {
                val vm: LoginViewModel = viewModel(
                    factory = LoginViewModelFactory(sdk.auth)
                )
                LoginScreen(
                    viewModel = vm,
                    onLoginSuccess = {
                        navController.navigate(Routes.HOME) {
                            popUpTo(Routes.LOGIN) { inclusive = true }
                        }
                    },
                    onGoToRegister = {
                        navController.navigate(Routes.REGISTER)
                    }
                )
            }

            //старый метод регистрации
            composable(Routes.REGISTER) {
                val vm: RegisterViewModel = viewModel(
                    factory = RegisterViewModelFactory(sdk.auth)
                )
                RegisterScreen(
                    viewModel = vm,
                    onRegisterSuccess = {
                        navController.navigate(Routes.HOME) {
                            popUpTo(Routes.LOGIN) { inclusive = true }
                        }
                    },
                    onGoToLogin = {
                        navController.popBackStack()
                    }
                )
            }
            /*navigation(
                route = Routes.REGISTER_GRAPH,
                startDestination = Routes.REGISTER_START
            ) {

                composable(Routes.REGISTER_START) { backStackEntry ->

                    val parentEntry = remember(backStackEntry) {
                        navController.getBackStackEntry(Routes.REGISTER_GRAPH)
                    }

                    val vm: RegisterViewModel = viewModel(
                        parentEntry,
                        factory = RegisterViewModelFactory(sdk.auth)
                    )

                    RegisterStartScreen(
                        viewModel = vm,
                        onNext = {
                            navController.navigate(Routes.REGISTER_BODY)
                        },
                        onGoToLogin = {
                            navController.popBackStack()
                        }
                    )
                }

                composable(Routes.REGISTER_BODY) { backStackEntry ->

                    val parentEntry = remember(backStackEntry) {
                        navController.getBackStackEntry(Routes.REGISTER_GRAPH)
                    }

                    val vm: RegisterViewModel = viewModel(
                        parentEntry,
                        factory = RegisterViewModelFactory(sdk.auth)
                    )

                    RegisterBodyScreen(
                        viewModel = vm,
                        onNext = {
                            navController.navigate(Routes.REGISTER_GOALS)
                        }
                    )
                }

                composable(Routes.REGISTER_GOALS) { backStackEntry ->

                    val parentEntry = remember(backStackEntry) {
                        navController.getBackStackEntry(Routes.REGISTER_GRAPH)
                    }

                    val vm: RegisterViewModel = viewModel(
                        parentEntry,
                        factory = RegisterViewModelFactory(sdk.auth)
                    )

                    RegisterGoalsScreen(
                        viewModel = vm,
                        onRegisterSuccess = {
                            startStepCountingWithPermission()

                            navController.navigate(Routes.HOME) {
                                popUpTo(Routes.LOGIN) { inclusive = true }
                            }
                        }
                    )
                }
            }*/


            composable(Routes.HOME) {
                val vm: HomeViewModel = viewModel(
                    factory = HomeViewModelFactory(sdk.activity, sdk.user)
                )
                HomeScreen(viewModel = vm)
            }

            composable(Routes.WORKOUTS) {
                val vm: WorkoutsViewModel = viewModel(
                    factory = WorkoutsViewModelFactory(sdk.workouts)
                )
                WorkoutsScreen(
                    viewModel = vm,
                    onWorkoutClick = { workoutId ->
                        navController.navigate("workouts/$workoutId")
                    }
                )
            }

            composable(Routes.WORKOUT_DETAIL) { backStackEntry ->
                val workoutId = backStackEntry.arguments?.getString("workoutId") ?: return@composable
                val vm: WorkoutDetailViewModel = viewModel(
                    factory = WorkoutDetailViewModelFactory(
                        workoutId = workoutId,
                        workoutRepo = sdk.workouts,
                        sessionRepo = sdk.sessions,
                        calculator = sdk.calorieCalculator,
                        userRepo = sdk.user
                    )
                )
                WorkoutDetailScreen(
                    viewModel = vm,
                    onBack = { navController.popBackStack() }
                )
            }

            composable(Routes.PROFILE) {
                val vm: ProfileViewModel = viewModel(
                    factory = ProfileViewModelFactory(sdk.user, sdk.auth)
                )
                ProfileScreen(
                    viewModel = vm,
                    onLogout = {
                        sdk.stopStepCounting()
                        navController.navigate(Routes.LOGIN) {
                            popUpTo(0) { inclusive = true }
                        }
                    },
                    onEnablePedometer = { startStepCountingWithPermission() },
                    onDisablePedometer = { sdk.stopStepCounting() }
                )
            }

            composable(Routes.FITVISION) {
                val vm: FitVisionViewModel = viewModel(
                    factory = FitVisionViewModelFactory(sdk.generation, sdk.user, sdk.activity)
                )
                FitVisionScreen(viewModel = vm)
            }
        }
    }
}
