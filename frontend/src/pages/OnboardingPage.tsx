import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import {
  answerQuiz,
  getRegisterItems,
  startQuiz,
  submitRegister,
  updateStoredUser,
  type QuizQuestion,
  type RegisterChoice,
  type RegisterCode,
  type RegisterItem,
  type RegisterItemsResponse,
} from "../api";

type Phase = "register" | "quiz-intro" | "quiz" | "done";

const DEPTH_TEXT: Record<string, string> = {
  easy: "We'll start with the basics and explain every term.",
  medium: "We'll start at a middle level and explain key terms as we go.",
  technical: "We'll start at a detailed level, using specialist terms freely.",
};

const btnPrimary =
  "rounded-lg bg-[#13294b] px-6 py-3 text-base font-semibold text-white shadow-md shadow-slate-900/10 transition hover:bg-[#0d1f3a] disabled:opacity-60";
const optionBtn =
  "flex w-full items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-left text-sm text-slate-700 transition hover:border-[#13294b] hover:bg-blue-50 disabled:opacity-60";

export default function OnboardingPage() {
  const navigate = useNavigate();
  const storedUser = JSON.parse(localStorage.getItem("user") || "null");

  const [phase, setPhase] = useState<Phase>(storedUser?.register ? "quiz-intro" : "register");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // language step
  const [meta, setMeta] = useState<RegisterItemsResponse | null>(null);
  const [items, setItems] = useState<RegisterItem[]>([]);
  const [step, setStep] = useState(0);
  const [choices, setChoices] = useState<RegisterChoice[]>([]);
  const [askDirect, setAskDirect] = useState(false);

  // quiz step
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [question, setQuestion] = useState<QuizQuestion | null>(null);
  const [depth, setDepth] = useState<string | null>(null);

  useEffect(() => {
    if (!localStorage.getItem("token")) {
      navigate("/login");
      return;
    }
    if (phase === "register") {
      getRegisterItems()
        .then((d) => {
          setMeta(d);
          setItems(d.items);
        })
        .catch((e) => setError((e as Error).message));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function registerDone(reg: RegisterCode) {
    updateStoredUser({ register: reg });
    setPhase("quiz-intro");
  }

  async function pick(reg: RegisterCode) {
    const next = [...choices.slice(0, step), { item_id: items[step].item_id, register: reg }];
    setChoices(next);
    if (next.length < 2) {
      setStep(step + 1);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const res = await submitRegister(next);
      if (res.done) registerDone(res.register);
      else if ("next_item" in res) {
        setItems([...items.slice(0, 2), res.next_item]);
        setStep(2);
      } else setAskDirect(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function chooseDirect(reg: RegisterCode) {
    setSaving(true);
    setError(null);
    try {
      const res = await submitRegister(choices, reg);
      if (res.done) registerDone(res.register);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function beginQuiz() {
    setSaving(true);
    setError(null);
    try {
      const res = await startQuiz();
      setSessionId(res.session_id);
      setQuestion(res.question);
      setPhase("quiz");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function answer(choice: number) {
    if (!sessionId || !question) return;
    setSaving(true);
    setError(null);
    try {
      const res = await answerQuiz(sessionId, question.question_id, choice);
      if (res.done) {
        setDepth(res.depth);
        updateStoredUser({ depth: res.depth, onboarding_completed: true });
        setPhase("done");
      } else {
        setQuestion(res.question);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  const sample = items[step];
  const card = "rounded-2xl border border-slate-200 bg-white p-6 shadow-sm";

  return (
    <div className="min-h-screen bg-[#e8eef6]">
      <Header minimal />

      <main className="mx-auto max-w-3xl px-6 py-10">
        <div className="mb-8">
          <h1 className="font-serif text-3xl font-semibold text-slate-900">Welcome to DiploMind</h1>
          <p className="mt-2 text-sm text-slate-600">
            {phase === "register"
              ? "First, let's find the language you're most comfortable reading."
              : "A few quick questions so we can match explanations to you."}
          </p>
        </div>

        {error && (
          <p role="alert" className="mb-6 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}{" "}
            {!localStorage.getItem("token") && (
              <button className="font-semibold underline" onClick={() => navigate("/login")}>
                Sign in again
              </button>
            )}
          </p>
        )}

        {/* ---------- language: samples ---------- */}
        {phase === "register" && !meta && !error && <p className="text-sm text-slate-600">Loading...</p>}

        {phase === "register" && meta && !askDirect && sample && (
          <section className={card}>
            <p className="mb-5 text-base font-medium text-slate-900">{meta.prompt.en}</p>
            <div className="space-y-2">
              {sample.options.map((opt, i) => {
                const rtl = opt.register === "ur";
                return (
                  <button key={opt.register} type="button" disabled={saving} onClick={() => pick(opt.register)} className={optionBtn}>
                    <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded bg-slate-200 text-xs font-semibold text-slate-600">
                      {String.fromCharCode(65 + i)}
                    </span>
                    <span dir={rtl ? "rtl" : "ltr"} className={`flex-1 leading-relaxed ${rtl ? "text-right text-base" : "text-left"}`}>
                      {opt.text}
                    </span>
                  </button>
                );
              })}
            </div>
          </section>
        )}

        {/* ---------- language: direct question ---------- */}
        {phase === "register" && meta && askDirect && (
          <section className={card}>
            <p className="mb-5 text-base font-medium text-slate-900">Which language would you like DiploMind to use?</p>
            <div className="grid gap-2 sm:grid-cols-3">
              {(Object.keys(meta.registers) as RegisterCode[]).map((reg) => (
                <button
                  key={reg}
                  type="button"
                  disabled={saving}
                  onClick={() => chooseDirect(reg)}
                  className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-base font-medium text-slate-800 transition hover:border-[#13294b] hover:bg-blue-50 disabled:opacity-60"
                >
                  {meta.registers[reg]}
                </button>
              ))}
            </div>
          </section>
        )}

        {/* ---------- quiz intro ---------- */}
        {phase === "quiz-intro" && (
          <section className={card}>
            <h2 className="text-xl font-semibold text-slate-900">A few questions on news and politics</h2>
            <p className="mt-2 text-slate-700">
              This helps us pitch explanations at the right level for you. There&apos;s no score.
            </p>
            <p className="mt-1 text-sm text-slate-500">
              If you&apos;re not sure of an answer, choose &quot;I don&apos;t know&quot; rather than guessing.
            </p>
            <button type="button" disabled={saving} onClick={beginQuiz} className={`mt-6 ${btnPrimary}`}>
              {saving ? "Starting..." : "Start"} <span aria-hidden="true">→</span>
            </button>
          </section>
        )}

        {/* ---------- quiz question ---------- */}
        {phase === "quiz" && question && (
          <section className={card} dir={question.lang === "ur" ? "rtl" : "ltr"}>
            <p className="mb-2 text-xs font-bold uppercase tracking-wider text-blue-950" dir="ltr">
              Question {question.number}
            </p>
            <p className={`mb-5 text-base font-medium text-slate-900 ${question.lang === "ur" ? "text-right" : ""}`}>
              {question.stem}
            </p>
            <div className="space-y-2">
              {question.options.map((opt, i) => {
                const idk = i === question.options.length - 1;
                return (
                  <button
                    key={`${question.question_id}-${i}`}
                    type="button"
                    disabled={saving}
                    onClick={() => answer(i)}
                    className={idk ? `${optionBtn} italic text-slate-500` : optionBtn}
                  >
                    {!idk && (
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded bg-slate-200 text-xs font-semibold text-slate-600">
                        {String.fromCharCode(65 + i)}
                      </span>
                    )}
                    <span className={`flex-1 leading-relaxed ${question.lang === "ur" ? "text-right" : "text-left"}`}>{opt}</span>
                  </button>
                );
              })}
            </div>
          </section>
        )}

        {/* ---------- done ---------- */}
        {phase === "done" && depth && (
          <section className={card}>
            <h2 className="text-xl font-semibold text-slate-900">You&apos;re all set!</h2>
            <p className="mt-2 text-slate-700">{DEPTH_TEXT[depth]}</p>
            <p className="mt-1 text-sm text-slate-500">DiploMind keeps adjusting as you learn.</p>
            <button type="button" onClick={() => navigate("/chat")} className={`mt-6 ${btnPrimary}`}>
              Start learning <span aria-hidden="true">→</span>
            </button>
          </section>
        )}
      </main>
    </div>
  );
}