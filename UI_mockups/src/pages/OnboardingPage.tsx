import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";

type Question = {
  id: string;
  text: string;
  options: string[];
  grid?: boolean;
  rtl?: number[];
};

type Section = { title: string; note: string; questions: Question[] };

const NOT_SURE = "I'm not sure";

const sections: Section[] = [
  {
    title: "How you like to read",
    note: "Pick whichever feels most natural. There are no wrong answers.",
    questions: [
      {
        id: "s1",
        text: "A news story about a ceasefire is being explained to you. Which version sounds most like you?",
        rtl: [1],
        options: [
          "Both sides have agreed to a temporary halt in fighting, with international mediators helping to broker the deal.",
          "دونوں فریقوں نے بین الاقوامی ثالثوں کی کوششوں سے عارضی جنگ بندی پر اتفاق کیا ہے۔",
          "Dono parties ne international mediators ki help se temporary ceasefire par agree kar liya hai.",
        ],
      },
      {
        id: "s2",
        text: "Which explanation of “economic sanctions” is the easiest for you to follow?",
        options: [
          "Sanctions are punishments one country uses on another, like blocking trade, so that it changes what it is doing.",
          "Sanctions are restrictions on trade or finance used by states to pressure another government to change its policy.",
          "Sanctions are coercive economic instruments, such as embargoes and asset freezes, used to impose costs on a target state's decision-makers.",
        ],
      },
      {
        id: "s3",
        text: "You come across an unfamiliar term like “non-alignment” in an article. What do you usually do?",
        options: [
          "Skip it and keep reading",
          "Want a short, everyday-language explanation",
          "Look for the full background and history",
        ],
      },
    ],
  },
  {
    title: "World affairs check",
    note: "Just a quick look at what you already know.",
    questions: [
      {
        id: "k1",
        text: "Which of these is a permanent member of the UN Security Council?",
        grid: true,
        options: ["China", "India", "Germany", "Pakistan", NOT_SURE],
      },
      {
        id: "k2",
        text: "What does “sovereignty” mean in international relations?",
        options: [
          "A state's supreme authority over its own territory",
          "A trade agreement between two countries",
          "A military alliance between several countries",
          NOT_SURE,
        ],
      },
      {
        id: "k3",
        text: "NATO is best described as:",
        grid: true,
        options: [
          "A military alliance",
          "A trade bloc",
          "A United Nations agency",
          NOT_SURE,
        ],
      },
      {
        id: "k4",
        text: "Which theory argues that states mainly act to increase their own power and security?",
        grid: true,
        options: ["Realism", "Liberalism", "Constructivism", NOT_SURE],
      },
      {
        id: "k5",
        text: "What is “soft power”?",
        options: [
          "Influence through culture, values and diplomacy",
          "Influence through military force",
          "Influence through economic sanctions",
          NOT_SURE,
        ],
      },
      {
        id: "k6",
        text: "What does a “multipolar world order” mean?",
        options: [
          "Power is spread among several major states",
          "One superpower dominates the world",
          "Two superpowers compete with each other",
          NOT_SURE,
        ],
      },
    ],
  },
];

const allQuestions = sections.flatMap((s) => s.questions);
const total = allQuestions.length;

export default function OnboardingPage() {
  const navigate = useNavigate();
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const answered = Object.keys(answers).length;
  const complete = answered === total;

  function numberOf(id: string) {
    return allQuestions.findIndex((q) => q.id === id) + 1;
  }

  return (
    <div className="min-h-screen bg-[#e8eef6]">
      <Header minimal />

      <main className="mx-auto max-w-3xl px-6 py-10">
        <div className="mb-8">
          <h1 className="font-serif text-3xl font-semibold text-slate-900">
            Welcome to DiploMind
          </h1>
          <p className="mt-2 text-sm text-slate-600">
            A few quick questions so we can match explanations to you.
          </p>

          <div className="mt-5 flex items-center gap-4">
            <div className="h-2 flex-1 rounded-full bg-slate-300/70">
              <div
                className="h-2 rounded-full bg-[#13294b] transition-all"
                style={{ width: `${(answered / total) * 100}%` }}
              />
            </div>
            <span className="text-sm font-medium text-slate-600">
              {answered} of {total}
            </span>
          </div>
        </div>

        <div className="space-y-10">
          {sections.map((section) => (
            <section key={section.title}>
              <div className="mb-4">
                <h2 className="text-xs font-bold uppercase tracking-wider text-blue-950">
                  {section.title}
                </h2>
                <p className="mt-1 text-sm text-slate-600">{section.note}</p>
              </div>

              <div className="space-y-4">
                {section.questions.map((q) => (
                  <fieldset
                    key={q.id}
                    className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
                  >
                    <legend className="sr-only">{q.text}</legend>
                    <div className="mb-4 flex gap-3">
                      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-blue-50 text-sm font-semibold text-[#13294b]">
                        {numberOf(q.id)}
                      </span>
                      <p className="pt-0.5 text-base font-medium text-slate-900">
                        {q.text}
                      </p>
                    </div>

                    <div
                      className={q.grid ? "grid gap-2 sm:grid-cols-2" : "space-y-2"}
                    >
                      {q.options.map((option, i) => {
                        const active = answers[q.id] === option;
                        const isRtl = q.rtl?.includes(i);
                        return (
                          <button
                            key={option}
                            type="button"
                            onClick={() =>
                              setAnswers((p) => ({ ...p, [q.id]: option }))
                            }
                            className={`flex w-full items-start gap-3 rounded-lg border px-4 py-3 text-left text-sm transition ${
                              active
                                ? "border-[#13294b] bg-blue-50 text-slate-900"
                                : "border-slate-200 bg-slate-50 text-slate-700 hover:border-slate-400 hover:bg-white"
                            }`}
                          >
                            <span
                              className={`flex h-6 w-6 shrink-0 items-center justify-center rounded text-xs font-semibold ${
                                active
                                  ? "bg-[#13294b] text-white"
                                  : "bg-slate-200 text-slate-600"
                              }`}
                            >
                              {String.fromCharCode(65 + i)}
                            </span>
                            {/* Only the text flips to RTL; the letter badge stays on the left */}
                            <span
                              dir={isRtl ? "rtl" : "ltr"}
                              className={`flex-1 leading-relaxed ${
                                isRtl ? "text-right" : "text-left"
                              }`}
                            >
                              {option}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </fieldset>
                ))}
              </div>
            </section>
          ))}
        </div>

        <div className="flex justify-end pb-6 pt-8">
          <button
            type="button"
            disabled={!complete}
            onClick={() => navigate("/chat")}
            className="rounded-lg bg-[#13294b] px-6 py-3 text-base font-semibold text-white shadow-md shadow-slate-900/10 transition hover:bg-[#0d1f3a] disabled:cursor-not-allowed disabled:opacity-50"
          >
            Finish and start learning <span aria-hidden="true">→</span>
          </button>
        </div>
      </main>
    </div>
  );
}