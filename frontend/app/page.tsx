import ApiStatus from "@/components/ApiStatus";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-8 px-6 py-16">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">Neural Debugger</h1>
        <p className="text-zinc-600 dark:text-zinc-400">
          Paste an error or stack trace and get a grounded explanation with citations from
          indexed documentation.
        </p>
      </header>
      <ApiStatus />
    </main>
  );
}
