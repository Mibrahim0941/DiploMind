import { useEffect, useRef, useState } from "react";
import Header from "../components/Header";

const history = [
  "NATO vs the United Nations",
  "What is soft power?",
  "Why do sanctions get imposed?",
  "Pakistan and the SCO",
];

type ChatInputProps = {
  /** Called when the user sends a message and/or attached documents. */
  onSend: (message: string, files: File[]) => void;
  placeholder?: string;
  disabled?: boolean;
};

const ACCEPTED = ".pdf,.doc,.docx,.txt,.md,.ppt,.pptx";

function formatSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function ChatInput({
  onSend,
  placeholder = "Ask about international relations…",
  disabled = false,
}: ChatInputProps) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);

  const canSend = !disabled && (text.trim().length > 0 || files.length > 0);

  function send() {
    if (!canSend) return;
    onSend(text.trim(), files);
    setText("");
    setFiles([]);
    if (textRef.current) textRef.current.style.height = "auto";
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends, Shift+Enter adds a new line
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function handleChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setText(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }

  function handleFiles(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = Array.from(e.target.files ?? []);
    if (picked.length) setFiles((prev) => [...prev, ...picked]);
    e.target.value = ""; // allow re-selecting the same file
  }

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }

  return (
    <div className="w-full rounded-2xl border border-slate-300 bg-white p-2 shadow-sm transition focus-within:border-blue-600 focus-within:ring-2 focus-within:ring-blue-600/20">
      {/* Attached documents */}
      {files.length > 0 && (
        <ul className="flex flex-wrap gap-2 px-2 pb-2 pt-1">
          {files.map((file, i) => (
            <li
              key={`${file.name}-${i}`}
              className="flex max-w-full items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 py-1.5 pl-3 pr-1.5 text-sm text-slate-700"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
                className="shrink-0 text-blue-700"
              >
                <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
                <path d="M14 3v5h5" />
              </svg>
              <span className="truncate">{file.name}</span>
              <span className="shrink-0 text-xs text-slate-400">
                {formatSize(file.size)}
              </span>
              <button
                type="button"
                onClick={() => removeFile(i)}
                aria-label={`Remove ${file.name}`}
                className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-slate-400 transition hover:bg-slate-200 hover:text-slate-700"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}

      <div className="flex items-end gap-2">
        {/* Upload button */}
        <input
          ref={fileRef}
          type="file"
          accept={ACCEPTED}
          multiple
          onChange={handleFiles}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          disabled={disabled}
          aria-label="Upload documents"
          title="Upload documents"
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-slate-500 transition hover:bg-slate-100 hover:text-[#13294b] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#13294b] disabled:cursor-not-allowed disabled:opacity-50"
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="m21.4 11.1-9.2 9.2a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5" />
          </svg>
        </button>

        {/* Message box */}
        <textarea
          ref={textRef}
          rows={1}
          value={text}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          className="max-h-40 min-h-10 flex-1 resize-none bg-transparent px-1 py-2 text-slate-900 placeholder-slate-400 outline-none disabled:opacity-60"
        />

        {/* Send button */}
        <button
          type="button"
          onClick={send}
          disabled={!canSend}
          aria-label="Send message"
          title="Send (Enter)"
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#13294b] text-white shadow-md shadow-slate-900/10 transition hover:-translate-y-0.5 hover:bg-[#0d1f3a] active:translate-y-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#13294b] focus-visible:ring-offset-2 disabled:translate-y-0 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none"
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M12 19V5" />
            <path d="m5 12 7-7 7 7" />
          </svg>
        </button>
      </div>
    </div>
  );
}

/* ------------------------------ Quiz popup ------------------------------ */

type QuizQuestion = {
  text: string;
  options: string[];
  answer: number; // index of the correct option
  explain: string;
};

// Questions based on the current conversation (NATO vs the UN)
const quiz: QuizQuestion[] = [
  {
    text: "Which best describes the United Nations?",
    options: [
      "A military alliance of Europe and North America",
      "A global organisation that almost every country belongs to",
      "A trade bloc for Asian countries",
    ],
    answer: 1,
    explain:
      "The UN is a worldwide organisation focused on keeping peace and helping countries cooperate.",
  },
  {
    text: "What do NATO members promise each other?",
    options: [
      "To share the same currency",
      "To follow every UN resolution",
      "To defend each other if one is attacked",
    ],
    answer: 2,
    explain:
      "NATO is a defence alliance: an attack on one member is treated as a concern for all.",
  },
  {
    text: "Where are most NATO members located?",
    options: [
      "Europe and North America",
      "Asia and Africa",
      "South America and Oceania",
    ],
    answer: 0,
    explain: "NATO's members are mostly in Europe and North America.",
  },
  {
    text: "True or false: NATO is an agency of the United Nations.",
    options: ["True", "False"],
    answer: 1,
    explain:
      "False. NATO is a separate military alliance; it is not part of the UN.",
  },
  {
    text: "Which statement sums up the difference best?",
    options: [
      "The UN is for the whole world and peace in general; NATO is a smaller group focused on joint defence",
      "NATO is for the whole world; the UN is a small defence group",
      "They are the same organisation with two names",
    ],
    answer: 0,
    explain:
      "The UN has near-universal membership and a broad peace mission, while NATO is a smaller alliance built around joint defence.",
  },
];

function QuizModal({ onClose }: { onClose: () => void }) {
  const [index, setIndex] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [score, setScore] = useState(0);
  const [finished, setFinished] = useState(false);

  // Close on Escape
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const q = quiz[index];
  const revealed = selected !== null;
  const isLast = index === quiz.length - 1;

  function choose(i: number) {
    if (revealed) return;
    setSelected(i);
    if (i === q.answer) setScore((s) => s + 1);
  }

  function next() {
    if (isLast) {
      setFinished(true);
    } else {
      setIndex((n) => n + 1);
      setSelected(null);
    }
  }

  function restart() {
    setIndex(0);
    setSelected(null);
    setScore(0);
    setFinished(false);
  }

  const verdict =
    score === quiz.length
      ? "Perfect score. You've got this topic."
      : score >= Math.ceil(quiz.length * 0.6)
        ? "Nice work. You understand the main ideas."
        : "Good start. Review the explanation in the chat and try again.";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Knowledge quiz"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-xl overflow-hidden rounded-2xl bg-white shadow-2xl"
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-200 bg-slate-50 px-6 py-4">
          <div>
            <p className="text-xs font-bold uppercase tracking-wider text-blue-950">
              Quick quiz
            </p>
            <p className="text-sm text-slate-600">NATO vs the United Nations</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close quiz"
            className="flex h-9 w-9 items-center justify-center rounded-lg text-xl text-slate-500 transition hover:bg-slate-200 hover:text-slate-800"
          >
            ×
          </button>
        </div>

        {finished ? (
          /* Results */
          <div className="px-6 py-10 text-center">
            <div className="mx-auto flex h-24 w-24 items-center justify-center rounded-full bg-blue-50 ring-4 ring-blue-100">
              <span className="font-serif text-3xl font-semibold text-[#13294b]">
                {score}/{quiz.length}
              </span>
            </div>
            <h2 className="mt-5 font-serif text-2xl font-semibold text-slate-900">
              {score === quiz.length ? "Excellent!" : "Quiz complete"}
            </h2>
            <p className="mx-auto mt-2 max-w-sm text-sm text-slate-600">
              {verdict}
            </p>
            <div className="mt-8 flex justify-center gap-3">
              <button
                type="button"
                onClick={restart}
                className="rounded-lg border border-slate-300 bg-white px-5 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
              >
                Retake quiz
              </button>
              <button
                type="button"
                onClick={onClose}
                className="rounded-lg bg-[#13294b] px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-slate-900/10 transition hover:bg-[#0d1f3a]"
              >
                Back to chat
              </button>
            </div>
          </div>
        ) : (
          /* Question */
          <div className="px-6 py-6">
            <div className="mb-5 flex items-center gap-4">
              <div className="h-2 flex-1 rounded-full bg-slate-200">
                <div
                  className="h-2 rounded-full bg-[#13294b] transition-all"
                  style={{
                    width: `${((index + (revealed ? 1 : 0)) / quiz.length) * 100}%`,
                  }}
                />
              </div>
              <span className="text-sm font-medium text-slate-600">
                {index + 1} of {quiz.length}
              </span>
            </div>

            <p className="mb-4 text-base font-medium text-slate-900">
              {q.text}
            </p>

            <div className="space-y-2">
              {q.options.map((option, i) => {
                const isCorrect = i === q.answer;
                const isPicked = i === selected;
                let style =
                  "border-slate-200 bg-slate-50 text-slate-700 hover:border-slate-400 hover:bg-white";
                let badge = "bg-slate-200 text-slate-600";
                if (revealed && isCorrect) {
                  style = "border-emerald-500 bg-emerald-50 text-slate-900";
                  badge = "bg-emerald-600 text-white";
                } else if (revealed && isPicked) {
                  style = "border-red-400 bg-red-50 text-slate-900";
                  badge = "bg-red-500 text-white";
                } else if (revealed) {
                  style = "border-slate-200 bg-slate-50 text-slate-400";
                }
                return (
                  <button
                    key={option}
                    type="button"
                    onClick={() => choose(i)}
                    disabled={revealed}
                    className={`flex w-full items-start gap-3 rounded-lg border px-4 py-3 text-left text-sm transition disabled:cursor-default ${style}`}
                  >
                    <span
                      className={`flex h-6 w-6 shrink-0 items-center justify-center rounded text-xs font-semibold ${badge}`}
                    >
                      {revealed && isCorrect
                        ? "✓"
                        : revealed && isPicked
                          ? "✕"
                          : String.fromCharCode(65 + i)}
                    </span>
                    <span className="leading-relaxed">{option}</span>
                  </button>
                );
              })}
            </div>

            {revealed && (
              <div
                role="status"
                className={`mt-4 rounded-lg px-4 py-3 text-sm leading-relaxed ${
                  selected === q.answer
                    ? "bg-emerald-50 text-emerald-900"
                    : "bg-amber-50 text-amber-900"
                }`}
              >
                <strong>
                  {selected === q.answer ? "Correct. " : "Not quite. "}
                </strong>
                {q.explain}
              </div>
            )}

            <div className="mt-6 flex justify-end">
              <button
                type="button"
                onClick={next}
                disabled={!revealed}
                className="rounded-lg bg-[#13294b] px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-slate-900/10 transition hover:bg-[#0d1f3a] disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isLast ? "See results" : "Next question"}{" "}
                <span aria-hidden="true">→</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatPageQuiz() {
  const [active, setActive] = useState(0);
  const [showQuiz, setShowQuiz] = useState(false);

  function handleSend(message: string, files: File[]) {
    // TODO: add the message to the conversation and upload `files` to your backend
    console.log("Send:", message, files);
  }

  return (
    <div className="flex h-screen flex-col bg-white">
      <Header />

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar: chat history */}
        <aside className="hidden w-72 shrink-0 flex-col border-r border-slate-200 bg-slate-50 md:flex">
          <div className="p-4">
            <button className="w-full rounded-lg bg-[#13294b] px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-[#0d1f3a]">
              + New chat
            </button>
          </div>
          <p className="px-5 pb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Recent
          </p>
          <nav className="flex-1 space-y-1 overflow-y-auto px-3 pb-4">
            {history.map((title, i) => (
              <button
                key={title}
                onClick={() => setActive(i)}
                className={`w-full truncate rounded-lg px-3 py-2.5 text-left text-sm transition ${
                  active === i
                    ? "bg-white font-medium text-slate-900 shadow-sm ring-1 ring-slate-200"
                    : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                {title}
              </button>
            ))}
          </nav>
        </aside>

        {/* Main chat */}
        <main className="flex min-w-0 flex-1 flex-col">
          <div className="flex-1 overflow-y-auto px-4 py-8 sm:px-8">
            <div className="mx-auto max-w-3xl space-y-6">
              {/* User message */}
              <div className="flex justify-end">
                <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-[#13294b] px-4 py-3 text-sm leading-relaxed text-white">
                  What is the difference between NATO and the United Nations?
                </div>
              </div>

              {/* Assistant message */}
              <div className="flex gap-3">
                <span className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-100 text-xs font-bold text-[#13294b]">
                  D
                </span>
                <div className="max-w-[85%]">
                  <div className="rounded-2xl rounded-tl-sm border border-slate-200 bg-slate-50 px-4 py-3 text-sm leading-relaxed text-slate-800">
                    <p>
                      The <strong>United Nations (UN)</strong> is a global
                      organisation that almost every country belongs to. Its
                      main goal is to keep world peace and help countries
                      cooperate.
                    </p>
                    <p className="mt-3">
                      <strong>NATO</strong> is a military alliance of some
                      countries, mostly in Europe and North America. Members
                      promise to defend each other if one of them is attacked.
                    </p>
                    <p className="mt-3">
                      In short: the UN is for the whole world and for peace in
                      general, while NATO is a smaller group focused on joint
                      defence.
                    </p>
                  </div>

                  {/* Quiz prompt */}
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-blue-100 bg-blue-50 px-4 py-3">
                    <p className="text-sm text-slate-700">
                      Want to check how much you took in?
                    </p>
                    <button
                      type="button"
                      onClick={() => setShowQuiz(true)}
                      className="rounded-lg bg-[#13294b] px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-[#0d1f3a]"
                    >
                      Take a 5-question quiz
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Input */}
          <div className="border-t border-slate-200 bg-white px-4 py-4 sm:px-8">
            <div className="mx-auto max-w-3xl">
              <div className="mb-2 flex justify-end">
                <button
                  type="button"
                  onClick={() => setShowQuiz(true)}
                  className="rounded-full border border-slate-300 bg-white px-3.5 py-1.5 text-xs font-semibold text-[#13294b] transition hover:border-[#13294b] hover:bg-blue-50"
                >
                  Test my knowledge
                </button>
              </div>
              <ChatInput
                onSend={handleSend}
                placeholder="Ask anything about international relations…"
              />
            </div>
          </div>
        </main>
      </div>

      {showQuiz && <QuizModal onClose={() => setShowQuiz(false)} />}
    </div>
  );
}