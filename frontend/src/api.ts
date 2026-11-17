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

// ---------- requests that need login ----------

async function authed<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${localStorage.getItem("token") ?? ""}`,
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (res.status === 401) {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
  }
  if (!res.ok) {
    throw new Error(
      typeof data.detail === "string" ? data.detail : "Something went wrong. Please try again.",
    );
  }
  return data as T;
}

export function updateStoredUser(patch: Partial<User>) {
  const stored = localStorage.getItem("user");
  if (stored) localStorage.setItem("user", JSON.stringify({ ...JSON.parse(stored), ...patch }));
}

// ---------- onboarding: language (register) ----------

export type RegisterCode = "en" | "ur" | "urdish";
export type RegisterItem = { item_id: string; options: { register: RegisterCode; text: string }[] };
export type RegisterChoice = { item_id: string; register: RegisterCode };
export type RegisterItemsResponse = {
  prompt: Record<RegisterCode, string>;
  registers: Record<RegisterCode, string>;
  items: RegisterItem[];
};
export type RegisterResult =
  | { done: true; register: RegisterCode }
  | { done: false; next_item: RegisterItem }
  | { done: false; ask_direct: true };

export function getRegisterItems() {
  return authed<RegisterItemsResponse>("GET", "/onboarding/register-items");
}

export function submitRegister(choices: RegisterChoice[], direct?: RegisterCode) {
  return authed<RegisterResult>("POST", "/onboarding/register", { choices, direct });
}

// ---------- onboarding: knowledge quiz ----------

export type QuizQuestion = {
  question_id: string;
  number: number;
  lang: RegisterCode;
  stem: string;
  options: string[]; // last one is "I don't know"
};
export type QuizStart = { session_id: string; question: QuizQuestion };
export type QuizAnswerResult =
  | { done: false; question: QuizQuestion }
  | { done: true; depth: string; confidence: number };

export function startQuiz() {
  return authed<QuizStart>("POST", "/onboarding/quiz/start", {});
}

export function answerQuiz(session_id: string, question_id: string, choice: number) {
  return authed<QuizAnswerResult>("POST", "/onboarding/quiz/answer", { session_id, question_id, choice });
}