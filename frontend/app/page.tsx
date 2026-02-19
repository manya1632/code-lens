import ReviewForm from "./components/ReviewForm";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-neutral-950 via-neutral-900 to-black text-white flex flex-col items-center px-6 pb-20">

      <div className="max-w-5xl w-full text-center mt-16">
        <h1 className="text-5xl font-extrabold bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent">
          CodeLens AI
        </h1>

        <p className="text-neutral-400 mt-4 text-lg">
          Deep Learning–based code review with bug detection, complexity analysis, and explainability.
        </p>

        <div className="mt-6 flex justify-center gap-4 text-sm text-neutral-400">
          <span className="px-3 py-1 border border-neutral-700 rounded-full">
            Supports: Python
          </span>
          <span className="px-3 py-1 border border-neutral-700 rounded-full">
            JavaScript
          </span>
          <span className="px-3 py-1 border border-neutral-700 rounded-full">
            Java
          </span>
        </div>
      </div>

      <ReviewForm/>
    </main>
  );
}
