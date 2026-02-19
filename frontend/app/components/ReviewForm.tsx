"use client";

import { useState } from "react";

export default function ReviewForm() {
  const [code, setCode] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    setResult(null);

    const res = await fetch(process.env.NEXT_PUBLIC_API_URL + "/review"
, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ code }),
    });

    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="w-full max-w-3xl mt-8">
      <textarea
        className="w-full h-48 p-4 bg-neutral-900 border border-neutral-700 rounded"
        placeholder="Paste your code here..."
        value={code}
        onChange={(e) => setCode(e.target.value)}
      />

      <button
        onClick={handleSubmit}
        className="mt-4 px-6 py-2 bg-blue-600 rounded hover:bg-blue-700"
      >
        {loading ? "Analyzing..." : "Analyze Code"}
      </button>

      {result && (
        <div className="mt-6 bg-neutral-900 p-4 rounded border border-neutral-700">
          <p><strong>Severity:</strong> {result.overall_severity}</p>
          <p><strong>Score:</strong> {result.score}</p>
          <p><strong>Complexity:</strong> {result.complexity_before}</p>
        </div>
      )}

    </div>
  );
}
