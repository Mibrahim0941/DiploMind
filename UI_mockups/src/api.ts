export type User = {
  id: string;
  name: string;
  email: string;
  register: string | null;
  depth: string | null;
  onboarding_completed: boolean;
};

type AuthResponse = { token: string; user: User };

async function post(path: string, body: unknown): Promise<AuthResponse> {
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(
      typeof data.detail === "string" ? data.detail : "Something went wrong. Please try again.",
    );
  }
  return data;
}

export function signup(body: { name: string; email: string; password: string; confirm_password: string }) {
  return post("/auth/signup", body);
}

export function login(body: { email: string; password: string }) {
  return post("/auth/login", body);
}

export function saveSession({ token, user }: AuthResponse) {
  localStorage.setItem("token", token);
  localStorage.setItem("user", JSON.stringify(user));
}