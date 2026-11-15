import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import logo from "../assets/logo.png";
import { saveSession, signup } from "../api";

const inputClass =
  "w-full rounded-lg border border-slate-300 bg-white px-4 py-3 text-slate-900 placeholder-slate-400 outline-none transition focus:border-blue-600 focus:ring-2 focus:ring-blue-600/20";

export default function SignupPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
    confirm: "",
  });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function update(field: keyof typeof form, value: string) {
    setForm((prev) => ({ ...prev, [field]: value }));
    setError(null);
  }

  async function handleSubmit(ev: React.SyntheticEvent<HTMLFormElement>) {
    ev.preventDefault();
    setError(null);

    // Same checks as the backend (use case 1), so the user sees errors instantly
    if (!form.name.trim() || !form.email.trim() || !form.password || !form.confirm) {
      setError("Please fill in all fields with valid values.");
      return;
    }
    if (form.password.length < 8 || form.password !== form.confirm) {
      setError("The password must be at least eight characters long and match the confirmation.");
      return;
    }

    setLoading(true);
    try {
      const res = await signup({
        name: form.name,
        email: form.email,
        password: form.password,
        confirm_password: form.confirm,
      });
      saveSession(res);
      navigate("/onboarding");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  function field(
    id: keyof typeof form,
    label: string,
    type: string,
    placeholder: string,
    autoComplete: string,
  ) {
    return (
      <div>
        <label
          htmlFor={`signup-${id}`}
          className="mb-1.5 block text-sm font-medium text-slate-700"
        >
          {label}
        </label>
        <input
          id={`signup-${id}`}
          type={type}
          autoComplete={autoComplete}
          value={form[id]}
          placeholder={placeholder}
          onChange={(ev) => update(id, ev.target.value)}
          className={inputClass}
        />
      </div>
    );
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
              Create account
            </h1>
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

            {field("name", "Full name", "text", "Your name", "name")}
            {field("email", "Email", "email", "name@example.com", "email")}
            {field(
              "password",
              "Password",
              "password",
              "At least 8 characters",
              "new-password",
            )}
            {field(
              "confirm",
              "Confirm password",
              "password",
              "Re-enter your password",
              "new-password",
            )}

            <button
              type="submit"
              disabled={loading}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-[#13294b] px-6 py-3 text-base font-semibold text-white shadow-md shadow-slate-900/10 transition hover:bg-[#0d1f3a] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#13294b] focus-visible:ring-offset-2 disabled:opacity-60"
            >
              {loading ? "Creating account..." : "Create account"}{" "}
              <span aria-hidden="true">→</span>
            </button>
          </form>

          <p className="mt-8 text-center text-sm text-slate-600">
            Already have an account?{" "}
            <Link
              to="/login"
              className="font-semibold text-blue-700 hover:text-blue-800 hover:underline"
            >
              Sign in
            </Link>
          </p>
        </div>
      </main>
    </div>
  );
}