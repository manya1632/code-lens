import ReviewForm from "./components/ReviewForm";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-neutral-950 via-neutral-900 to-black text-white flex flex-col items-center p-6">

      <div className="max-w-5xl w-full text-center mt-10">
        <h1 className="text-5xl font-extrabold bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent">
          CodeLens AI
        </h1>

        <p className="text-neutral-400 mt-4 text-lg">
          Deep Learning–based code review with bug detection, complexity analysis, and explainability.
        </p>
      </div>

      <ReviewForm />
    </main>
  );
}
