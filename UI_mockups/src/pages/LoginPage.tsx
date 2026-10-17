import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

// Put your logo file in the /public folder (e.g. public/logo.png) and
// reference it with an absolute path. Change this if your file is named differently.
import logo from "../assets/logo.png";

export default function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!email.trim() || !password) {
      setError("Please enter your email and password.");
      return;
    }
    navigate("/chat");
  }

  return (
    <div className="grid min-h-screen bg-slate-50 lg:grid-cols-2">
      {/* Left: brand panel */}
      <aside
        className="relative hidden flex-col overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-blue-900 px-12 py-12 text-white lg:flex"
        style={{
          backgroundImage:
            "linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(to bottom right, #0f172a, #1e293b, #1e3a8a)",
          backgroundSize: "40px 40px, 40px 40px, 100% 100%",
        }}
      >
        {/* Logo + title (mt-10 lowers it; adjust to taste) */}
        <div className="mt-10 flex items-center gap-4">
          <img
            src={logo}
            alt="DiploMind logo"
            className="h-16 w-16 shrink-0 object-contain"
          />
          <span className="text-5xl font-semibold leading-none tracking-wide">
            DiploMind
          </span>
        </div>

        {/* Headline block, vertically centered in the remaining space */}
        <div className="flex flex-1 flex-col justify-center">
          <h2 className="font-serif text-4xl font-semibold leading-tight">
            World affairs, explained
            <br />
            the way you understand them.
          </h2>
          <p className="mt-5 max-w-md text-base leading-relaxed text-slate-300">
            An adaptive platform for learning international relations, built for
            Pakistani learners and tuned to your language and level.
          </p>
        </div>
      </aside>

      {/* Right: form */}
      <main className="flex items-center justify-center px-6 py-12 sm:px-12">
        <div className="w-full max-w-md">
          <div className="mb-8 flex items-center gap-3 lg:hidden">
            <img
              src={logo}
              alt="DiploMind logo"
              className="h-10 w-10 shrink-0 object-contain"
            />
            <span className="text-2xl font-semibold text-slate-900">
              DiploMind
            </span>
          </div>

          <header className="mb-8">
            <h1 className="font-serif text-3xl font-semibold text-slate-900">
              Sign in
            </h1>
            <p className="mt-2 text-sm text-slate-600">
              Continue your learning on DiploMind.
            </p>
          </header>

          <form onSubmit={handleSubmit} className="space-y-5" noValidate>
            {error && (
              <p
                role="alert"
                className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
              >
                {error}
              </p>
            )}

            <div>
              <label
                htmlFor="login-email"
                className="mb-1.5 block text-sm font-medium text-slate-700"
              >
                Email
              </label>
              <input
                id="login-email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  setError(null);
                }}
                placeholder="name@example.com"
                className="w-full rounded-lg border border-slate-300 bg-white px-4 py-3 text-slate-900 placeholder-slate-400 outline-none transition focus:border-blue-600 focus:ring-2 focus:ring-blue-600/20"
              />
            </div>

            <div>
              <label
                htmlFor="login-password"
                className="mb-1.5 block text-sm font-medium text-slate-700"
              >
                Password
              </label>
              <input
                id="login-password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  setError(null);
                }}
                placeholder="••••••••"
                className="w-full rounded-lg border border-slate-300 bg-white px-4 py-3 text-slate-900 placeholder-slate-400 outline-none transition focus:border-blue-600 focus:ring-2 focus:ring-blue-600/20"
              />
            </div>

            <button
              type="submit"
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-[#13294b] px-6 py-3 text-base font-semibold text-white shadow-md shadow-slate-900/10 transition hover:bg-[#0d1f3a] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#13294b] focus-visible:ring-offset-2"
            >
              Sign in <span aria-hidden="true">→</span>
            </button>
          </form>

          <p className="mt-8 text-center text-sm text-slate-600">
            Don&apos;t have an account?{" "}
            <Link
              to="/signup"
              className="font-semibold text-blue-700 hover:text-blue-800 hover:underline"
            >
              Create account
            </Link>
          </p>
        </div>
      </main>
    </div>
  );
}
