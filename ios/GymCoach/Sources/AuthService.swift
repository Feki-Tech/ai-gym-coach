// Sign in — Apple, Google or Microsoft.
//
// Conventions: the same OpenID Connect flow as the desktop (`coach_auth.py`):
// Authorization Code + PKCE (S256), `state` and `nonce`, the system browser
// via ASWebAuthenticationSession (never an embedded web view), provider
// discovery, ID-token claim checks (iss / aud / exp / nonce). App Store
// Review Guideline 4.8 requires an equivalent Apple option whenever a
// third-party login is offered, so Sign in with Apple is always there and
// listed first. Identity is stored in the Keychain; nothing about training
// is sent to any provider.

import Foundation
import AuthenticationServices
import Combine
import CryptoKit
import Security
import UIKit

struct AthleteIdentity: Codable, Equatable {
    let provider: String      // "apple" | "google" | "microsoft"
    let subject: String
    let email: String
    let name: String
    let signedIn: Date
}

enum AuthError: LocalizedError {
    case notConfigured(String)
    case cancelled
    case invalidResponse(String)
    case tokenRejected(String)

    var errorDescription: String? {
        switch self {
        case .notConfigured(let p): return "\(p) sign-in is not configured for this build."
        case .cancelled: return "Sign-in cancelled."
        case .invalidResponse(let s): return "Sign-in failed: \(s)"
        case .tokenRejected(let s): return "The identity token was rejected: \(s)"
        }
    }
}

/// Client IDs come from Info.plist (set in project.yml / at build time), never
/// from source. Empty = the provider's button is hidden.
enum AuthConfig {
    static var googleClientID: String { plist("GoogleClientID") }
    static var microsoftClientID: String { plist("MicrosoftClientID") }
    static var microsoftTenant: String { plist("MicrosoftTenant").isEmpty ? "common" : plist("MicrosoftTenant") }
    static var bundleID: String { Bundle.main.bundleIdentifier ?? "tech.fekitech.gymcoach" }

    private static func plist(_ key: String) -> String {
        (Bundle.main.object(forInfoDictionaryKey: key) as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    /// Google iOS clients redirect to the reversed client id as a scheme.
    static var googleRedirect: (scheme: String, uri: String)? {
        let id = googleClientID
        guard id.hasSuffix(".apps.googleusercontent.com") else { return nil }
        let scheme = "com.googleusercontent.apps." + String(id.dropLast(".apps.googleusercontent.com".count))
        return (scheme, scheme + ":/oauth2redirect")
    }

    /// Microsoft's documented iOS redirect for public clients.
    static var microsoftRedirect: (scheme: String, uri: String) {
        ("msauth.\(bundleID)", "msauth.\(bundleID)://auth")
    }
}

// MARK: - PKCE / random

enum PKCE {
    static func randomURLSafe(_ bytes: Int = 32) -> String {
        var buf = [UInt8](repeating: 0, count: bytes)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes, &buf)
        return base64URL(Data(buf))
    }

    static func challenge(for verifier: String) -> String {
        base64URL(Data(SHA256.hash(data: Data(verifier.utf8))))
    }

    static func base64URL(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    static func base64URLDecode(_ s: String) -> Data? {
        var str = s.replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
        while str.count % 4 != 0 { str += "=" }
        return Data(base64Encoded: str)
    }
}

// MARK: - Keychain

enum Keychain {
    private static let service = "tech.fekitech.gymcoach.identity"

    static func save(_ identity: AthleteIdentity) throws {
        let data = try JSONEncoder().encode(identity)
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                    kSecAttrService as String: service,
                                    kSecAttrAccount as String: "athlete"]
        SecItemDelete(query as CFDictionary)
        var add = query
        add[kSecValueData as String] = data
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let status = SecItemAdd(add as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw AuthError.invalidResponse("keychain error \(status)")
        }
    }

    static func load() -> AthleteIdentity? {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                    kSecAttrService as String: service,
                                    kSecAttrAccount as String: "athlete",
                                    kSecReturnData as String: true,
                                    kSecMatchLimit as String: kSecMatchLimitOne]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return nil }
        return try? JSONDecoder().decode(AthleteIdentity.self, from: data)
    }

    static func clear() {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                    kSecAttrService as String: service]
        SecItemDelete(query as CFDictionary)
    }
}

// MARK: - OIDC helpers

struct OIDCProvider {
    let name: String
    let clientID: String
    let discoveryURL: URL
    let redirectScheme: String
    let redirectURI: String
    let issuerPrefix: String

    static func google() -> OIDCProvider? {
        guard let r = AuthConfig.googleRedirect else { return nil }
        return OIDCProvider(name: "google", clientID: AuthConfig.googleClientID,
                            discoveryURL: URL(string: "https://accounts.google.com/.well-known/openid-configuration")!,
                            redirectScheme: r.scheme, redirectURI: r.uri,
                            issuerPrefix: "https://accounts.google.com")
    }

    static func microsoft() -> OIDCProvider? {
        let id = AuthConfig.microsoftClientID
        guard !id.isEmpty else { return nil }
        let r = AuthConfig.microsoftRedirect
        return OIDCProvider(name: "microsoft", clientID: id,
                            discoveryURL: URL(string: "https://login.microsoftonline.com/\(AuthConfig.microsoftTenant)/v2.0/.well-known/openid-configuration")!,
                            redirectScheme: r.scheme, redirectURI: r.uri,
                            issuerPrefix: "https://login.microsoftonline.com/")
    }
}

/// Decoded, claim-checked ID token. The signature is guaranteed by the TLS
/// channel to the provider's token endpoint (OIDC Core §3.1.3.7 allows this
/// for the code flow); claims are still checked so a wrong token is refused.
struct IDToken {
    let claims: [String: Any]

    init(_ token: String, provider: OIDCProvider, nonce: String) throws {
        let parts = token.split(separator: ".")
        guard parts.count == 3, let payload = PKCE.base64URLDecode(String(parts[1])),
              let obj = try? JSONSerialization.jsonObject(with: payload) as? [String: Any]
        else { throw AuthError.tokenRejected("malformed") }
        guard let iss = obj["iss"] as? String,
              iss == provider.issuerPrefix || iss.hasPrefix(provider.issuerPrefix)
                || (provider.name == "google" && iss == "accounts.google.com")
        else { throw AuthError.tokenRejected("issuer") }
        let aud = (obj["aud"] as? [String]) ?? [(obj["aud"] as? String) ?? ""]
        guard aud.contains(provider.clientID) else { throw AuthError.tokenRejected("audience") }
        guard let exp = obj["exp"] as? Double, exp > Date().timeIntervalSince1970 - 120
        else { throw AuthError.tokenRejected("expired") }
        guard (obj["nonce"] as? String) == nonce else { throw AuthError.tokenRejected("nonce") }
        claims = obj
    }

    var identity: AthleteIdentity? {
        guard let sub = claims["sub"] as? String else { return nil }
        let email = (claims["email"] as? String) ?? (claims["preferred_username"] as? String) ?? ""
        let name = (claims["name"] as? String) ?? String(email.split(separator: "@").first ?? "")
        return AthleteIdentity(provider: "", subject: sub, email: email, name: name, signedIn: Date())
    }
}

// MARK: - Service

@MainActor
final class AuthService: NSObject, ObservableObject {
    static let shared = AuthService()

    @Published private(set) var identity: AthleteIdentity? = Keychain.load()
    @Published private(set) var busy = false
    @Published var lastError: String?

    private var session: ASWebAuthenticationSession?
    private var appleNonce: String?
    private var appleContinuation: CheckedContinuation<AthleteIdentity, Error>?

    var googleAvailable: Bool { OIDCProvider.google() != nil }
    var microsoftAvailable: Bool { OIDCProvider.microsoft() != nil }

    func signOut() {
        Keychain.clear()
        identity = nil
    }

    // MARK: Google / Microsoft (OIDC code + PKCE in the system browser)

    func signIn(with provider: OIDCProvider) async {
        busy = true
        defer { busy = false }
        lastError = nil
        do {
            let config = try await discover(provider)
            let verifier = PKCE.randomURLSafe(48)
            let state = PKCE.randomURLSafe(24)
            let nonce = PKCE.randomURLSafe(24)
            var comps = URLComponents(url: config.authorizationEndpoint, resolvingAgainstBaseURL: false)!
            comps.queryItems = [
                .init(name: "client_id", value: provider.clientID),
                .init(name: "redirect_uri", value: provider.redirectURI),
                .init(name: "response_type", value: "code"),
                .init(name: "scope", value: "openid email profile"),
                .init(name: "state", value: state),
                .init(name: "nonce", value: nonce),
                .init(name: "code_challenge", value: PKCE.challenge(for: verifier)),
                .init(name: "code_challenge_method", value: "S256"),
                .init(name: "prompt", value: "select_account"),
            ]
            let callback = try await browserFlow(url: comps.url!, scheme: provider.redirectScheme)
            let q = URLComponents(url: callback, resolvingAgainstBaseURL: false)?.queryItems ?? []
            func item(_ n: String) -> String? { q.first { $0.name == n }?.value }
            guard item("state") == state else { throw AuthError.invalidResponse("state mismatch") }
            if let err = item("error") { throw AuthError.invalidResponse(item("error_description") ?? err) }
            guard let code = item("code") else { throw AuthError.invalidResponse("no code") }
            var req = URLRequest(url: config.tokenEndpoint)
            req.httpMethod = "POST"
            req.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
            req.httpBody = [
                "grant_type=authorization_code",
                "code=\(code.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? code)",
                "redirect_uri=\(provider.redirectURI.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? provider.redirectURI)",
                "client_id=\(provider.clientID)",
                "code_verifier=\(verifier)",
            ].joined(separator: "&").data(using: .utf8)
            let (data, resp) = try await URLSession.shared.data(for: req)
            guard (resp as? HTTPURLResponse)?.statusCode == 200,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let idToken = json["id_token"] as? String
            else { throw AuthError.invalidResponse("token endpoint") }
            let token = try IDToken(idToken, provider: provider, nonce: nonce)
            guard let base = token.identity else { throw AuthError.tokenRejected("no subject") }
            let ident = AthleteIdentity(provider: provider.name, subject: base.subject,
                                        email: base.email, name: base.name, signedIn: Date())
            try Keychain.save(ident)
            identity = ident
        } catch {
            lastError = (error as? AuthError)?.errorDescription ?? error.localizedDescription
        }
    }

    private struct Discovery { let authorizationEndpoint: URL; let tokenEndpoint: URL }

    private func discover(_ p: OIDCProvider) async throws -> Discovery {
        let (data, _) = try await URLSession.shared.data(from: p.discoveryURL)
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let a = (obj["authorization_endpoint"] as? String).flatMap(URL.init(string:)),
              let t = (obj["token_endpoint"] as? String).flatMap(URL.init(string:))
        else { throw AuthError.invalidResponse("discovery") }
        return Discovery(authorizationEndpoint: a, tokenEndpoint: t)
    }

    private func browserFlow(url: URL, scheme: String) async throws -> URL {
        try await withCheckedThrowingContinuation { cont in
            let s = ASWebAuthenticationSession(url: url, callbackURLScheme: scheme) { cb, err in
                if let cb { cont.resume(returning: cb) }
                else if let e = err as? ASWebAuthenticationSessionError, e.code == .canceledLogin {
                    cont.resume(throwing: AuthError.cancelled)
                } else { cont.resume(throwing: err ?? AuthError.invalidResponse("browser")) }
            }
            s.presentationContextProvider = self
            s.prefersEphemeralWebBrowserSession = false   // keep the user's provider session
            self.session = s
            if !s.start() { cont.resume(throwing: AuthError.invalidResponse("could not open browser")) }
        }
    }

    // MARK: Sign in with Apple

    func signInWithApple() async {
        busy = true
        defer { busy = false }
        lastError = nil
        let raw = PKCE.randomURLSafe(32)
        appleNonce = raw
        let request = ASAuthorizationAppleIDProvider().createRequest()
        request.requestedScopes = [.fullName, .email]
        // Apple's convention: the request carries the SHA-256 hex of the raw nonce
        request.nonce = SHA256.hash(data: Data(raw.utf8)).map { String(format: "%02x", $0) }.joined()
        let controller = ASAuthorizationController(authorizationRequests: [request])
        controller.delegate = self
        controller.presentationContextProvider = self
        do {
            let ident: AthleteIdentity = try await withCheckedThrowingContinuation { cont in
                appleContinuation = cont
                controller.performRequests()
            }
            try Keychain.save(ident)
            identity = ident
        } catch {
            lastError = (error as? AuthError)?.errorDescription ?? error.localizedDescription
        }
    }
}

extension AuthService: ASAuthorizationControllerDelegate {
    nonisolated func authorizationController(controller: ASAuthorizationController,
                                             didCompleteWithAuthorization authorization: ASAuthorization) {
        Task { @MainActor in
            guard let cred = authorization.credential as? ASAuthorizationAppleIDCredential else {
                appleContinuation?.resume(throwing: AuthError.invalidResponse("apple credential"))
                appleContinuation = nil
                return
            }
            // Apple returns name/e-mail only on the first authorization; keep
            // whatever we already have for later sign-ins.
            let previous = Keychain.load()
            let name = [cred.fullName?.givenName, cred.fullName?.familyName]
                .compactMap { $0 }.joined(separator: " ")
            let ident = AthleteIdentity(
                provider: "apple", subject: cred.user,
                email: cred.email ?? previous?.email ?? "",
                name: name.isEmpty ? (previous?.name ?? "") : name,
                signedIn: Date())
            appleContinuation?.resume(returning: ident)
            appleContinuation = nil
        }
    }

    nonisolated func authorizationController(controller: ASAuthorizationController,
                                             didCompleteWithError error: Error) {
        Task { @MainActor in
            if let e = error as? ASAuthorizationError, e.code == .canceled {
                appleContinuation?.resume(throwing: AuthError.cancelled)
            } else {
                appleContinuation?.resume(throwing: error)
            }
            appleContinuation = nil
        }
    }
}

extension AuthService: ASWebAuthenticationPresentationContextProviding,
                       ASAuthorizationControllerPresentationContextProviding {
    nonisolated func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
        Self.keyWindow()
    }

    nonisolated func presentationAnchor(for controller: ASAuthorizationController) -> ASPresentationAnchor {
        Self.keyWindow()
    }

    nonisolated private static func keyWindow() -> ASPresentationAnchor {
        MainActor.assumeIsolated {
            let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            return scenes.flatMap(\.windows).first { $0.isKeyWindow } ?? ASPresentationAnchor()
        }
    }
}
