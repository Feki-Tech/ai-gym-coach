import SwiftUI
import AuthenticationServices

/// Account screen: Sign in with Apple (always, per App Store guideline 4.8),
/// Google and Microsoft when a client id is configured for the build.
struct AccountView: View {
    @ObservedObject private var auth = AuthService.shared
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        List {
            if let id = auth.identity {
                Section {
                    HStack(spacing: 14) {
                        Image(systemName: "person.crop.circle.fill")
                            .font(.system(size: 44))
                            .foregroundStyle(.tint)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(id.name.isEmpty ? id.email : id.name).font(.headline)
                            if !id.email.isEmpty {
                                Text(id.email).font(.subheadline).foregroundStyle(.secondary)
                            }
                            Text(String(format: NSLocalizedString("via %@", comment: ""),
                                        providerName(id.provider)))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                    Button(role: .destructive) {
                        auth.signOut()
                    } label: {
                        Label("Sign out", systemImage: "rectangle.portrait.and.arrow.right")
                    }
                } header: {
                    Text("Signed in")
                } footer: {
                    Text("Your name and e-mail are kept in this iPhone's keychain and shown in your summaries. Nothing about your training is sent to the provider.")
                }
            } else {
                Section {
                    SignInWithAppleButton(.signIn) { _ in } onCompletion: { _ in }
                        .signInWithAppleButtonStyle(scheme == .dark ? .white : .black)
                        .frame(height: 46)
                        .overlay {                       // route through the service
                            Color.clear.contentShape(Rectangle())
                                .onTapGesture { Task { await auth.signInWithApple() } }
                        }
                        .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 4, trailing: 16))
                    if let g = OIDCProvider.google() {
                        providerButton("Sign in with Google", icon: "g.circle.fill") {
                            await auth.signIn(with: g)
                        }
                    }
                    if let m = OIDCProvider.microsoft() {
                        providerButton("Sign in with Microsoft", icon: "m.square.fill") {
                            await auth.signIn(with: m)
                        }
                    }
                    if auth.busy {
                        HStack { ProgressView(); Text("Signing in…").foregroundStyle(.secondary) }
                    }
                    if let err = auth.lastError {
                        Text(err).font(.footnote).foregroundStyle(.red)
                    }
                } header: {
                    Text("Sign in")
                } footer: {
                    Text("Optional. Signing in puts your name on your summaries and lets the same account open your progress dashboard. The sign-in page is the provider's own, in a secure system browser sheet; only your name and e-mail are read.")
                }
                if !auth.googleAvailable && !auth.microsoftAvailable {
                    Section {
                        Text("Google and Microsoft sign-in appear when a client id is configured for this build (docs/AUTH.md).")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                }
            }
        }
        .navigationTitle("Account")
    }

    private func providerButton(_ title: LocalizedStringKey, icon: String,
                                action: @escaping () async -> Void) -> some View {
        Button {
            Task { await action() }
        } label: {
            Label(title, systemImage: icon)
                .frame(maxWidth: .infinity, minHeight: 30)
        }
        .buttonStyle(.bordered)
        .disabled(auth.busy)
        .listRowInsets(EdgeInsets(top: 4, leading: 16, bottom: 4, trailing: 16))
    }

    private func providerName(_ p: String) -> String {
        switch p {
        case "apple": return "Apple"
        case "google": return "Google"
        case "microsoft": return "Microsoft"
        default: return p
        }
    }
}
