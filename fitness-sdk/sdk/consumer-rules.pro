# Keep all public SDK classes so consumers don't need their own rules
-keep public class com.app.fitness.** { public *; }
-keep interface com.app.fitness.** { *; }
-keep enum com.app.fitness.** { *; }

# Retrofit / OkHttp
-keepattributes Signature, InnerClasses, EnclosingMethod
-keepattributes RuntimeVisibleAnnotations, RuntimeVisibleParameterAnnotations
-keepclassmembers,allowshrinking,allowobfuscation interface * {
    @retrofit2.http.* <methods>;
}
-dontwarn org.codehaus.moshi.**
-dontwarn javax.annotation.**

# Gson
-keepattributes *Annotation*
-keep class com.app.fitness.http.dto.** { *; }
-keep class com.google.gson.** { *; }
