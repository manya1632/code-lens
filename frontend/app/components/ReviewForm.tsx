"use client";

import { useState, useEffect } from "react";

const SAMPLE_CODES: Record<string, string> = {
  Logic: `def divide(a, b):
    return a / b

print(divide(10, 0))`,

  Performance: `def find_duplicates(arr):
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i != j and arr[i] == arr[j]:
                print(arr[i])`,

  Security: `const query = "SELECT * FROM users WHERE id = " + userInput;
db.execute(query);`,

  Concurrency: `let counter = 0;
function increment() {
  counter++;
}`,

  Style: `def foo():
    x=1
    y=2
    print(x+y)`
};

export default function ReviewForm() {
  const [code, setCode] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState<string>("Unknown");

  // Auto detect language
  useEffect(() => {
  const detectLanguage = (code: string) => {
    const c = code.trim();

    // Python patterns
    if (
      c.includes("def ") ||
      c.includes("print(") ||
      c.includes("import ") ||
      c.includes("elif ") ||
      c.includes("None") ||
      c.includes("self")
    ) {
      return "Python";
    }

    // JavaScript patterns
    if (
      c.includes("function ") ||
      c.includes("console.log") ||
      c.includes("=>") ||
      c.includes("const ") ||
      c.includes("let ") ||
      c.includes("var ")
    ) {
      return "JavaScript";
    }

    // Java patterns
    if (
      c.includes("public class") ||
      c.includes("System.out.println") ||
      c.includes("public static void main") ||
      c.includes("import java")
    ) {
      return "Java";
    }

    return "Unknown";
  };

  setLanguage(detectLanguage(code));
}, [code]);


  const handleSubmit = async () => {
    if (!code.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const res = await fetch(
        process.env.NEXT_PUBLIC_API_URL + "/review",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ code }),
        }
      );

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
    }

    setLoading(false);
  };

  const severityColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case "low":
        return "bg-green-600";
      case "medium":
        return "bg-yellow-600";
      case "high":
        return "bg-red-600";
      default:
        return "bg-neutral-600";
    }
  };

  return (
    <div className="w-full max-w-5xl mt-16 space-y-10">

      {/* Sample Buttons */}
      <div>
        <h2 className="text-lg font-semibold mb-3 text-neutral-300">
          Try Sample Bugs
        </h2>
        <div className="flex flex-wrap gap-3">
          {Object.keys(SAMPLE_CODES).map((type) => (
            <button
              key={type}
              onClick={() => setCode(SAMPLE_CODES[type])}
              className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 rounded-lg text-sm transition"
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* Editor */}
      <div className="bg-neutral-900/60 border border-neutral-800 rounded-2xl p-6 shadow-xl">

        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold">Code Editor</h2>
          <span className="px-3 py-1 text-xs bg-neutral-800 border border-neutral-700 rounded-full">
            Language: {language}
          </span>
        </div>

        <textarea
          className="w-full h-56 p-4 bg-neutral-950 border border-neutral-700 rounded-lg text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-600"
          placeholder="Paste your code here..."
          value={code}
          onChange={(e) => setCode(e.target.value)}
        />

        <button
          onClick={handleSubmit}
          className="mt-6 px-6 py-3 bg-blue-600 rounded-lg font-semibold hover:bg-blue-700 transition-all shadow-lg"
        >
          {loading ? "Analyzing..." : "Analyze Code"}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-8">

          {/* Severity */}
          <div className="bg-neutral-900/60 border border-neutral-800 rounded-2xl p-6">
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-semibold">Overall Severity</h2>
              <span className={`px-4 py-1 rounded-full text-sm font-bold ${severityColor(result.overall_severity)}`}>
                {result.overall_severity.toUpperCase()}
              </span>
            </div>
            <p className="text-neutral-400 mt-2">
              Confidence: {Math.round(result.severity_score * 100)}%
            </p>
          </div>

          {/* Issues */}
          <div>
            <h2 className="text-xl font-semibold mb-4">Detected Issues</h2>
            <div className="grid gap-4">
              {result.issues?.map((issue: any, i: number) => (
                <div key={i} className="bg-neutral-900 border border-neutral-800 rounded-xl p-5">
                  <div className="flex justify-between">
                    <h3 className="font-semibold text-blue-400">
                      {issue.bug_type.toUpperCase()}
                    </h3>
                    <span className="text-sm text-neutral-400">
                      {(issue.confidence * 100).toFixed(0)}%
                    </span>
                  </div>

                  <p className="mt-3 text-neutral-300">
                    {issue.description}
                  </p>

                  <div className="mt-3 p-3 bg-neutral-800 rounded-lg text-sm text-neutral-400">
                    💡 {issue.suggestion}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Complexity */}
          <div className="bg-neutral-900/60 border border-neutral-800 rounded-2xl p-6">
            <h2 className="text-xl font-semibold mb-3">Complexity Analysis</h2>
            <div className="flex gap-10 text-neutral-300">
              <div>
                <p className="text-neutral-500 text-sm">Before</p>
                <p className="text-white">{result.complexity_before}</p>
              </div>
              <div>
                <p className="text-neutral-500 text-sm">After</p>
                <p className="text-white">{result.complexity_after}</p>
              </div>
            </div>
          </div>

          {/* Explanation */}
          <div className="bg-neutral-900/60 border border-neutral-800 rounded-2xl p-6">
            <h2 className="text-xl font-semibold mb-3">Model Explanation</h2>
            <p className="text-neutral-400 whitespace-pre-line">
              {result.explanation}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
