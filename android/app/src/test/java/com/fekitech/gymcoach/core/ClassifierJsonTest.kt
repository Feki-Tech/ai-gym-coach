// The classifier file arrives by adb push — untrusted input. A malformed
// or shape-inconsistent model must load as null (rule tier keeps working),
// never as an object whose first predict() crashes mid-workout.
// SECURITY.md S14.
package com.fekitech.gymcoach.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class ClassifierJsonTest {

    private val valid = """
        {"classes": ["a", "b"], "min_proba": 0.5,
         "W1": [[0.1, 0.2], [0.3, 0.4]], "b1": [0.0, 0.0],
         "W2": [[1.0, 0.0], [0.0, 1.0]], "b2": [0.0, 0.0],
         "mu": [0.0, 0.0], "sd": [1.0, 1.0],
         "manifest": {"model_version": "v9"}}
    """.trimIndent()

    @Test fun validModelLoadsAndPredicts() {
        val m = tinyMlpFromJson(valid)
        assertNotNull(m)
        assertEquals("v9", m!!.modelVersion)
        assertEquals(2, m.inputDim)
        val p = m.predict(doubleArrayOf(1.0, 0.0))
        assertEquals(1.0, p.sum(), 1e-9)
    }

    @Test fun garbageIsNull() {
        assertNull(tinyMlpFromJson("not json at all"))
        assertNull(tinyMlpFromJson("{}"))
        assertNull(tinyMlpFromJson("""{"classes": []}"""))
    }

    @Test fun shapeMismatchesAreNull() {
        // sd shorter than mu
        assertNull(tinyMlpFromJson(valid.replace(
            "\"sd\": [1.0, 1.0]", "\"sd\": [1.0]")))
        // ragged W2 row (wrong class count)
        assertNull(tinyMlpFromJson(valid.replace(
            "[0.0, 1.0]]", "[0.0]]")))
        // W1 rows don't match hidden size
        assertNull(tinyMlpFromJson(valid.replace(
            "[0.3, 0.4]", "[0.3]")))
        // empty classes vs 2-wide W2
        assertNull(tinyMlpFromJson(valid.replace(
            """["a", "b"]""", """["a"]""")))
    }

    @Test fun poisonedValuesAreNull() {
        // a zero sd would divide by zero on every frame
        assertNull(tinyMlpFromJson(valid.replace(
            "\"sd\": [1.0, 1.0]", "\"sd\": [1.0, 0.0]")))
        // NaN weights would make every probability NaN
        assertNull(tinyMlpFromJson(valid.replace(
            "\"b1\": [0.0, 0.0]", "\"b1\": [\"NaN\", 0.0]")))
    }
}
