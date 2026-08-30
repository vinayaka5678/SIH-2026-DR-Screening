package com.sih2026.drscreening.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext

private val LightColorScheme = lightColorScheme(
    primary = Primary,
    secondary = Secondary,
    surface = SurfaceLight,
    background = BackgroundLight,
    onPrimary = OnPrimaryLight,
    onSecondary = OnSecondaryLight
)

private val DarkColorScheme = darkColorScheme(
    primary = Primary,
    secondary = Secondary,
    surface = SurfaceDark,
    background = BackgroundDark,
    onPrimary = OnPrimaryDark,
    onSecondary = OnSecondaryDark
)

@Composable
fun DRScreeningTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}
