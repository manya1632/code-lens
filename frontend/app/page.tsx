import ReviewForm from "./components/ReviewForm";

export default function Home() {
  return (
    <main className="min-h-screen bg-neutral-950 text-white flex flex-col items-center p-6">
      <h1 className="text-4xl font-bold">CodeLens AI</h1>
      <ReviewForm/>
    </main>
  );
}
