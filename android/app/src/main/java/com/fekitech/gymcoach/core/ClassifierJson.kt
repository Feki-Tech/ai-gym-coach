// JSON loading for the portable classifier, kept out of Classifier.kt so
// the numeric core stays free of the org.json dependency (Android ships
// org.json in the framework; JVM unit tests pull it as a test dependency).
package com.fekitech.gymcoach.core

import org.json.JSONObject

fun tinyMlpFromJson(text: String): TinyMlp {
    val d = JSONObject(text)
    fun vec(key: String): DoubleArray {
        val a = d.getJSONArray(key)
        return DoubleArray(a.length()) { a.getDouble(it) }
    }
    fun mat(key: String): Array<DoubleArray> {
        val a = d.getJSONArray(key)
        return Array(a.length()) { i ->
            val row = a.getJSONArray(i)
            DoubleArray(row.length()) { row.getDouble(it) }
        }
    }
    val cls = d.getJSONArray("classes")
    return TinyMlp(
        classes = List(cls.length()) { cls.getString(it) },
        minProba = d.optDouble("min_proba", 0.75),
        w1 = mat("W1"), b1 = vec("b1"),
        w2 = mat("W2"), b2 = vec("b2"),
        mu = vec("mu"), sd = vec("sd"),
        modelVersion = d.optJSONObject("manifest")
            ?.optString("model_version", "unknown") ?: "unknown",
    )
}
