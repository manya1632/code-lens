"use client";

import { useState } from "react";

export default function ReviewForm() {
  const [code, setCode] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

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
    <div className="w-full max-w-5xl mt-12">

      {/* Editor Card */}
      <div className="bg-neutral-900/60 backdrop-blur-lg border border-neutral-800 rounded-2xl p-6 shadow-xl">
        <textarea
          className="w-full h-56 p-4 bg-neutral-950 border border-neutral-700 rounded-lg text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-600"
          placeholder="Paste your code here..."
          value={code}
          onChange={(e) => setCode(e.target.value)}
        />

        <button
          onClick={handleSubmit}
          className="mt-6 px-6 py-3 bg-blue-600 rounded-lg font-semibold hover:bg-blue-700 transition-all duration-200 shadow-lg"
        >
          {loading ? "Analyzing..." : "Analyze Code"}
        </button>
      </div>

      {/* Result Section */}
      {result && (
        <div className="mt-10 space-y-8">

          {/* Severity */}
          <div className="bg-neutral-900/60 border border-neutral-800 rounded-2xl p-6 shadow-lg">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Overall Severity</h2>
              <span
                className={`px-4 py-1 rounded-full text-sm font-bold ${severityColor(
                  result.overall_severity
                )}`}
              >
                {result.overall_severity.toUpperCase()}
              </span>
            </div>

            <p className="text-neutral-400 mt-2">
              Confidence Score: {Math.round(result.severity_score * 100)}%
            </p>
          </div>

          {/* Issues */}
          <div>
            <h2 className="text-xl font-semibold mb-4">Detected Issues</h2>

            <div className="grid gap-4">
              {result.issues?.map((issue: any, i: number) => (
                <div
                  key={i}
                  className="bg-neutral-900/60 border border-neutral-800 rounded-xl p-5 hover:border-blue-600 transition-all"
                >
                  <div className="flex justify-between items-center">
                    <h3 className="font-semibold text-blue-400">
                      {issue.bug_type}
                    </h3>
                    <span className="text-sm text-neutral-400">
                      {(issue.confidence * 100).toFixed(0)}%
                    </span>
                  </div>

                  <p className="mt-2 text-neutral-300">
                    {issue.description}
                  </p>

                  <p className="mt-2 text-sm text-neutral-500">
                    💡 {issue.suggestion}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Complexity */}
          <div className="bg-neutral-900/60 border border-neutral-800 rounded-2xl p-6 shadow-lg">
            <h2 className="text-xl font-semibold mb-2">Complexity Analysis</h2>

            <p className="text-neutral-400">
              Before: <span className="text-white">{result.complexity_before}</span>
            </p>
            <p className="text-neutral-400">
              After: <span className="text-white">{result.complexity_after}</span>
            </p>
          </div>

          {/* Explanation */}
          <div className="bg-neutral-900/60 border border-neutral-800 rounded-2xl p-6 shadow-lg">
            <h2 className="text-xl font-semibold mb-2">Model Explanation</h2>
            <p className="text-neutral-400 whitespace-pre-line">
              {result.explanation}
            </p>
          </div>

        </div>
      )}
    </div>
  );
}
