import Image from "next/image";

export default function SponsorBar() {
  return (
    <footer className="relative shrink-0 border-t border-gray-200/90 bg-white/92 dark:border-cyan-400/30 dark:bg-slate-900">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-500/55 to-transparent dark:via-cyan-300/90" />
      <div className="mx-auto flex w-full max-w-7xl items-center justify-center px-3 py-2 sm:px-5">
        <div className="flex w-full items-center justify-center rounded-xl border border-gray-200/80 bg-gradient-to-r from-slate-50 via-white to-cyan-50/70 px-2 py-1.5 shadow-[0_-6px_24px_rgba(15,23,42,0.05)] dark:border-slate-700 dark:bg-slate-800 dark:shadow-[0_-10px_28px_rgba(2,6,23,0.45)]">
          <div className="flex w-full items-center justify-center rounded-lg border border-slate-200 bg-white px-2 py-1 shadow-[inset_0_1px_0_rgba(255,255,255,0.85),0_6px_18px_rgba(15,23,42,0.08)] dark:border-cyan-200/70 dark:bg-[linear-gradient(180deg,rgba(255,255,255,1),rgba(241,245,249,0.98))] dark:shadow-[0_8px_24px_rgba(8,47,73,0.22)]">
            <div className="relative h-7 w-full max-w-4xl sm:h-8">
              <Image
                src="/sponsors/bannerIA3.png"
                alt="Colaboradores y patrocinadores"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 1024px"
                priority
              />
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
