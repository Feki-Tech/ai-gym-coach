// Client for coach_server.py — the desktop's LLM coach (persona, retrieval
// over the knowledge base and exercise catalogue, guardrails, history and
// profile tools) reached over the local network. Paired once with a URL
// and a code; every request carries the code as a bearer token.

import Foundation
import Combine

struct CoachAction: Equatable {
    let `do`: String
    let args: [String: Any]

    static func == (a: CoachAction, b: CoachAction) -> Bool { a.do == b.do }

    var exercise: String? { args["exercise"] as? String }
    var reps: Int? { (args["reps"] as? NSNumber)?.intValue }
    var seconds: Double? { (args["seconds"] as? NSNumber)?.doubleValue }
    var eccentricS: Double? { (args["eccentric_s"] as? NSNumber)?.doubleValue }
    var enabled: Bool? { args["enabled"] as? Bool }
    var kg: Double? { (args["kg"] as? NSNumber)?.doubleValue }
    var plan: String? { args["plan"] as? String }
}

enum CoachEvent {
    case delta(String)
    case action(CoachAction, ack: String)
    case done(reply: String)
    case error(String)
}

struct CoachHealth: Decodable {
    struct LLM: Decodable { let model: String; let reachable: Bool; let error: String }
    struct Knowledge: Decodable { let chunks: Int; let exercises: Int }
    let llm: LLM
    let prompt_version: String
    let knowledge: Knowledge
}

@MainActor
final class CoachClient: ObservableObject {
    static let shared = CoachClient()

    @Published var baseURL: String {
        didSet { UserDefaults.standard.set(baseURL, forKey: "coach.url") }
    }
    @Published var token: String {
        didSet { UserDefaults.standard.set(token, forKey: "coach.token") }
    }
    @Published private(set) var health: CoachHealth?
    @Published private(set) var lastError: String?

    private init() {
        baseURL = UserDefaults.standard.string(forKey: "coach.url") ?? ""
        token = UserDefaults.standard.string(forKey: "coach.token") ?? ""
    }

    var isConfigured: Bool { url(for: "/health") != nil && !token.isEmpty }

    private func url(for path: String, query: [String: String] = [:]) -> URL? {
        var base = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !base.isEmpty else { return nil }
        if !base.hasPrefix("http") { base = "http://" + base }
        guard var comps = URLComponents(string: base) else { return nil }
        comps.path = path
        if !query.isEmpty {
            comps.queryItems = query.map { URLQueryItem(name: $0.key, value: $0.value) }
        }
        return comps.url
    }

    private func request(_ path: String, method: String = "GET", body: [String: Any]? = nil,
                         query: [String: String] = [:]) throws -> URLRequest {
        guard let u = url(for: path, query: query) else {
            throw URLError(.badURL)
        }
        var req = URLRequest(url: u)
        req.httpMethod = method
        req.timeoutInterval = 120
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        if let body {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        return req
    }

    // MARK: - plain calls

    @discardableResult
    func checkHealth() async -> Bool {
        do {
            let (data, resp) = try await URLSession.shared.data(for: try request("/health"))
            guard let code = (resp as? HTTPURLResponse)?.statusCode else { throw URLError(.badServerResponse) }
            if code == 401 { throw CoachClientError.unauthorized }
            health = try JSONDecoder().decode(CoachHealth.self, from: data)
            lastError = nil
            return true
        } catch {
            health = nil
            lastError = (error as? CoachClientError)?.message ?? error.localizedDescription
            return false
        }
    }

    func brief(exercise: String?) async -> [String: Any]? {
        guard isConfigured else { return nil }
        var q: [String: String] = [:]
        if let e = exercise { q["exercise"] = e }
        guard let req = try? request("/brief", query: q),
              let (data, _) = try? await URLSession.shared.data(for: req),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              !obj.isEmpty else { return nil }
        return obj
    }

    /// Append a finished set to the desktop workout log (dashboard, history).
    func upload(recordJSON: Data) async -> Bool {
        guard isConfigured,
              let obj = try? JSONSerialization.jsonObject(with: recordJSON) as? [String: Any],
              let req = try? request("/log", method: "POST", body: ["record": obj]),
              let (_, resp) = try? await URLSession.shared.data(for: req)
        else { return false }
        return (resp as? HTTPURLResponse)?.statusCode == 200
    }

    // MARK: - streaming

    func chat(_ text: String, state: [String: Any]) -> AsyncStream<CoachEvent> {
        stream(path: "/chat", body: ["text": text, "state": state])
    }

    func event(_ event: String, payload: [String: Any], state: [String: Any]) -> AsyncStream<CoachEvent> {
        stream(path: "/event", body: ["event": event, "payload": payload, "state": state])
    }

    private func stream(path: String, body: [String: Any]) -> AsyncStream<CoachEvent> {
        AsyncStream { cont in
            let task = Task {
                do {
                    let req = try request(path, method: "POST", body: body)
                    let (bytes, resp) = try await URLSession.shared.bytes(for: req)
                    let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
                    if code == 401 { cont.yield(.error(CoachClientError.unauthorized.message)); cont.finish(); return }
                    if code != 200 { cont.yield(.error("HTTP \(code)")); cont.finish(); return }
                    for try await line in bytes.lines {
                        guard line.hasPrefix("data: "),
                              let data = line.dropFirst(6).data(using: .utf8),
                              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                        else { continue }
                        if let err = obj["error"] as? String { cont.yield(.error(err)) }
                        if let d = obj["delta"] as? String { cont.yield(.delta(d)) }
                        if let a = obj["action"] as? [String: Any], let d = a["do"] as? String {
                            cont.yield(.action(CoachAction(do: d, args: a), ack: obj["ack"] as? String ?? ""))
                        }
                        if obj["done"] as? Bool == true {
                            cont.yield(.done(reply: obj["reply"] as? String ?? ""))
                        }
                    }
                } catch {
                    if !Task.isCancelled { cont.yield(.error(error.localizedDescription)) }
                }
                cont.finish()
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }
}

enum CoachClientError: Error {
    case unauthorized
    var message: String {
        switch self {
        case .unauthorized: return NSLocalizedString("Pairing code rejected — check the code printed by coach_server.py.", comment: "")
        }
    }
}
