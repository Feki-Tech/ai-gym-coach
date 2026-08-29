// The classifier file is dropped into the app's documents — untrusted
// input. A malformed or shape-inconsistent model must load as nil (rule
// tier keeps working), never as an object whose first predict() crashes
// mid-workout. SECURITY.md S14.

import XCTest
@testable import CoachCore

final class ClassifierLoadTests: XCTestCase {

    private let valid = """
        {"classes": ["a", "b"], "min_proba": 0.5,
         "W1": [[0.1, 0.2], [0.3, 0.4]], "b1": [0.0, 0.0],
         "W2": [[1.0, 0.0], [0.0, 1.0]], "b2": [0.0, 0.0],
         "mu": [0.0, 0.0], "sd": [1.0, 1.0],
         "manifest": {"model_version": "v9"}}
        """

    private func load(_ s: String) -> TinyMLP? {
        TinyMLP.load(json: Data(s.utf8))
    }

    func testValidModelLoadsAndPredicts() {
        guard let m = load(valid) else {
            XCTFail("valid model refused")
            return
        }
        XCTAssertEqual(m.modelVersion, "v9")
        let p = m.predict([1.0, 0.0])
        XCTAssertEqual(p.reduce(0, +), 1.0, accuracy: 1e-9)
    }

    func testGarbageIsNil() {
        XCTAssertNil(load("not json at all"))
        XCTAssertNil(load("{}"))
        XCTAssertNil(load(#"{"classes": []}"#))
    }

    func testShapeMismatchesAreNil() {
        // sd shorter than mu
        XCTAssertNil(load(valid.replacingOccurrences(
            of: #""sd": [1.0, 1.0]"#, with: #""sd": [1.0]"#)))
        // ragged W2 row (wrong class count)
        XCTAssertNil(load(valid.replacingOccurrences(
            of: "[0.0, 1.0]]", with: "[0.0]]")))
        // W1 rows don't match hidden size
        XCTAssertNil(load(valid.replacingOccurrences(
            of: "[0.3, 0.4]", with: "[0.3]")))
        // 1 class vs 2-wide W2
        XCTAssertNil(load(valid.replacingOccurrences(
            of: #"["a", "b"]"#, with: #"["a"]"#)))
        // a string where a weight should be: vector() drops it -> short row
        XCTAssertNil(load(valid.replacingOccurrences(
            of: #""b1": [0.0, 0.0]"#, with: #""b1": [0.0, "x"]"#)))
    }

    func testPoisonedValuesAreNil() {
        // a zero sd would divide by zero on every frame
        XCTAssertNil(load(valid.replacingOccurrences(
            of: #""sd": [1.0, 1.0]"#, with: #""sd": [1.0, 0.0]"#)))
    }
}
