"use client";

import React from "react";
import { X, BookOpenText, ChevronRight, Info, ListChecks, MousePointerClick } from "lucide-react";

type HowToOverlayProps = {
  open: boolean;
  onClose: () => void;
};

type Section = {
  key: string;
  title: string;
  content: React.ReactNode;
};

function SectionItem({
  open,
  title,
  onToggle,
  children,
  id,
}: {
  open: boolean;
  title: React.ReactNode;
  onToggle: () => void;
  children: React.ReactNode;
  id: string;
}) {
  return (
    <div className="rounded-xl border border-base bg-card">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={onToggle}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onToggle();
          }
        }}
        className="group flex w-full items-center gap-2 px-3 py-2 text-left transition hover:bg-input focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300 rounded-xl"
      >
        <ChevronRight
          className={`h-4 w-4 shrink-0 transition-transform ${open ? "rotate-90 opacity-80" : "opacity-50"}`}
          aria-hidden="true"
        />
        <span className="text-sm font-medium">{title}</span>
      </button>

      {/* Collapsible body with smooth animation */}
      <div
        id={id}
        role="region"
        aria-hidden={!open}
        className={`grid transition-all duration-300 ease-out ${
          open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
        }`}
      >
        <div className="overflow-hidden">
          <div className="px-4 pb-3 pt-1 text-sm leading-6">{children}</div>
        </div>
      </div>
    </div>
  );
}

export default function HowToOverlay({ open, onClose }: HowToOverlayProps) {
  // ESC zamyka
  React.useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const sections: Section[] = [
    {
      key: "start",
      title: "Szybki start (60 sekund)",
      content: (
        <div className="space-y-2">
          <ol className="list-decimal space-y-1 pl-6">
            <li>Zaloguj się lub załóż konto z menu w prawym górnym rogu.</li>
            <li>
              Opisz objawy naturalnym językiem lub po przecinku, np.: <em>„gorączka, ból gardła, suchy kaszel”</em>.
            </li>
            <li>
              Po co najmniej <strong>3 potwierdzonych</strong> objawach system pokaże <strong>Top-3</strong> możliwych przyczyn.
            </li>
            <li>
              Użyj szybkich akcji (przyciski pod czatem): <em>Choroba?</em>, <em>Zalecenia</em>, <em>Pilność</em>, <em>Leki OTC</em>,
              <em> Profilaktyka</em>, <em>Podsumuj</em>.
            </li>
            <li>
              Jeśli coś się zmieniło, dopisz po prostu nową wiadomość, np. <em>„jednak nie mam gorączki”</em>, a system zaktualizuje stan.
            </li>
          </ol>
          <div className="mt-2 flex items-start gap-2 rounded-lg border border-base bg-input/50 p-2 text-xs">
            <MousePointerClick className="mt-[2px] h-4 w-4 opacity-70" />
            <p>
              Pod koniec rozmowy możesz poprosić: <em>„Podsumuj moje objawy i zalecenia w punktach”</em>, a dostaniesz czytelną listę.
            </p>
          </div>
        </div>
      ),
    },
    {
      key: "objawy",
      title: "Wprowadzanie i korygowanie objawów",
      content: (
        <div className="space-y-3">
          <ul className="list-disc space-y-1 pl-6">
            <li>
              System mapuje słowa na wewnętrzny słownik objawów (synonimy po polsku są rozpoznawane). Przykłady:
              <em> „katar = nieżyt nosa”, „duszność = skrócony oddech”</em>.
            </li>
            <li>
              Negacje działają: <em>„nie mam gorączki”, „brak kaszlu”, „to nie ból ucha”</em>; przestawiają stan na „nieobecny”.
            </li>
            <li>
              Doprecyzowanie jest mile widziane: czas trwania (<em>„od 3 dni”</em>), okoliczności (<em>„po wysiłku”</em>), pora dnia,
              czynniki ryzyka. To pomaga w lepszej ocenie.
            </li>
            <li>
              Zmiana zdania? Napisz po prostu nową wiadomość, np.: <em>„jednak mam stan podgorączkowy ~37.7°C”</em>.
            </li>
          </ul>
          <div className="rounded-lg border border-base bg-input/40 p-3">
            <div className="mb-1 flex items-center gap-2 text-sm font-medium">
              <ListChecks className="h-4 w-4 opacity-70" /> Dobre przykłady wpisów
            </div>
            <ul className="list-disc space-y-1 pl-6 text-sm">
              <li>
                „<em>Silny ból gardła od wczoraj wieczorem, suchy kaszel, stan podgorączkowy 37.6</em>”
              </li>
              <li>
                „<em>Katar wodnisty, kichanie, <strong>brak</strong> bólu głowy</em>”
              </li>
              <li>
                „<em>Kaszel nasila się w nocy, bez krwi, duszność przy wchodzeniu po schodach</em>”
              </li>
            </ul>
          </div>
        </div>
      ),
    },
    {
      key: "pytania",
      title: "Pytania uzupełniające (Tak / Nie)",
      content: (
        <div className="space-y-2">
          <p>
            Aplikacja zadaje krótkie pytania o kluczowe objawy (np. <em>„Czy masz niedrożność nosa?”</em>). Najlepsza odpowiedź to
            jednoznaczne <strong>„Tak”</strong> lub <strong>„Nie”</strong>. Dopuszczalne są krótkie warianty typu
            <em> „raczej tak”</em>, <em>„raczej nie”</em> też zostaną zinterpretowane.
          </p>
          <ul className="list-disc space-y-1 pl-6">
            <li>Jeśli nie jesteś pewien/pewna, napisz wprost: „nie wiem” lub opisz krótko wątpliwość.</li>
            <li>Możesz dopisać szczegóły: „Tak, od 2 dni”, „Nie, tylko rano bywa lekka niedrożność”.</li>
          </ul>
        </div>
      ),
    },
    {
      key: "predykcja",
      title: "Predykcje i Top-3 (XGBoost + River)",
      content: (
        <div className="space-y-3">
          <p>
            Silnik korzysta z dwóch komplementarnych modeli:
            <br />
            <strong>XGBoost</strong> (uczenie wsadowe, stabilne na dużym zbiorze) oraz{" "}
            <strong>River</strong> (uczenie przyrostowe, dobrze reaguje na zmiany rozkładów).
            Wynik końcowy to fuzja (ensemble), która prezentuje <strong>Top-3</strong> najbardziej prawdopodobne przyczyny.
          </p>
          <div className="rounded-xl border border-base bg-input/60 p-3 text-xs">
            <div className="flex items-center gap-2 font-medium">
              <Info className="h-4 w-4 opacity-70" /> Jak czytać Top-3?
            </div>
            <ul className="mt-1 list-disc space-y-1 pl-6">
              <li><strong>#1</strong>: diagnoza najbardziej zgodna z obecnymi objawami.</li>
              <li><strong>#2–#3</strong>: sensowne alternatywy; rozważ je, jeśli pojawią się nowe symptomy.</li>
              <li>
                Udziały procentowe mają charakter probabilistyczny z kalibracją. To nie jest pewnik, tylko wskazówka kliniczna.
              </li>
            </ul>
          </div>
          <p className="text-xs text-muted">
            Uwaga: modele nie uczą się automatycznie z Twojej pojedynczej rozmowy. Aktualizacje odbywają się okresowo na bazie
            zweryfikowanych danych, aby zachować spójność i bezpieczeństwo.
          </p>
        </div>
      ),
    },
    {
      key: "sterowanie",
      title: "Sterowanie rozmową i przydatne komendy",
      content: (
        <div className="space-y-2">
          <ul className="list-disc space-y-1 pl-6">
            <li>
              Zakończenie zbierania: napisz <em>„to wszystko”</em> / <em>„koniec”</em> / <em>„podsumuj”</em>.
            </li>
            <li>
              Korekta: napisz naturalnie, np. <em>„jednak nie mam gorączki”</em> albo <em>„dodaj: ból mięśni”</em>, a system zaktualizuje stan.
            </li>
            <li>
              Szybkie akcje pod czatem uruchamiają gotowe odpowiedzi: <em>Choroba?</em>, <em>Zalecenia</em>, <em>Pilność</em> itd.
            </li>
            <li>
              Klawiatura: <strong>Esc</strong> zamyka okna (pomoc, ustawienia). <strong>Enter</strong> wysyła wiadomość.
            </li>
          </ul>
        </div>
      ),
    },
    {
      key: "historia",
      title: "Historia rozmów, konto i ustawienia",
      content: (
        <div className="space-y-2">
          <ul className="list-disc space-y-1 pl-6">
            <li>
              <strong>Historia czatów</strong>: w menu użytkownika; pozwala wrócić do poprzednich sesji i prześledzić decyzje.
            </li>
            <li>
              <strong>Ustawienia konta</strong>: uzupełnij imię/nazwisko i datę urodzenia; komunikaty (np. o pilności) mogą być dzięki temu bardziej adekwatne.
            </li>
            <li>
              Prywatność: rozmowy są przechowywane w Twoim koncie. Możesz je usuwać z historii, gdy uznasz to za potrzebne.
            </li>
          </ul>
        </div>
      ),
    },
    {
      key: "zastrzezenia",
      title: "Ważne zastrzeżenia i bezpieczeństwo",
      content: (
        <div className="space-y-2">
          <p>
            Narzędzie wspiera <strong>wstępną</strong> ocenę objawów i edukację zdrowotną. Nie zastępuje badania lekarskiego.
          </p>
          <ul className="list-disc space-y-1 pl-6">
            <li>W przypadku objawów alarmowych lub nagłego pogorszenia stanu skontaktuj się z lekarzem lub dzwoń na 112/999.</li>
            <li>Treści mają charakter informacyjny i nie stanowią porady medycznej.</li>
          </ul>
        </div>
      ),
    },
  ];

  // Zarządzanie otwartymi sekcjami
  const [openKeys, setOpenKeys] = React.useState<Set<string>>(() => new Set(["start"]));
  const isOpen = (key: string) => openKeys.has(key);
  const toggle = (key: string) =>
    setOpenKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  const expandAll = () => setOpenKeys(new Set(sections.map((s) => s.key)));
  const collapseAll = () => setOpenKeys(new Set());

  if (!open) return null;

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 z-50 bg-black/50" onClick={onClose} aria-hidden="true" />

      {/* Modal */}
      <section
        role="dialog"
        aria-modal="true"
        aria-label="Jak używać aplikacji"
        className="fixed inset-0 z-50 grid place-items-center p-4"
        onClick={(e) => {
          if (e.target === e.currentTarget) onClose();
        }}
      >
        <div className="relative w-full max-w-3xl overflow-hidden rounded-2xl border border-base bg-card/95 backdrop-blur-md shadow-2xl">
          {/* Header */}
          <header className="flex items-center gap-2 border-b border-base bg-card/80 px-4 py-2">
            <span className="grid h-8 w-8 place-items-center rounded-full border border-base bg-input">
              <BookOpenText className="h-4 w-4 opacity-80" />
            </span>
            <h2 className="text-sm font-semibold">Jak używać aplikacji</h2>
            <div className="ml-auto flex items-center gap-2">
              <button
                onClick={expandAll}
                className="rounded-lg border border-base bg-input px-2 py-1 text-xs transition hover:brightness-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
                title="Rozwiń wszystko"
              >
                Rozwiń wszystko
              </button>
              <button
                onClick={collapseAll}
                className="rounded-lg border border-base bg-input px-2 py-1 text-xs transition hover:brightness-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
                title="Zwiń wszystko"
              >
                Zwiń wszystko
              </button>
              <button
                onClick={onClose}
                className="rounded-lg border border-base bg-input p-1.5 text-xs transition hover:brightness-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
                aria-label="Zamknij"
                title="Zamknij"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </header>

          {/* Aurora accent bar */}
          <div className="relative h-[3px] w-full overflow-hidden">
            <div
              className="h-full w-full"
              style={{ background: "linear-gradient(90deg, #34d399, #38bdf8 55%, #a78bfa)" }}
            />
            <div
              aria-hidden
              className="pointer-events-none absolute left-0 top-0 h-full w-1/3 opacity-30"
              style={{
                background: "linear-gradient(90deg, transparent, white, transparent)",
                animation: "auroraSlide 2.4s linear infinite",
              }}
            />
          </div>

          {/* Body */}
          <div className="max-h-[78vh] overflow-y-auto p-4 text-sm leading-6 space-y-2">
            {sections.map((s) => (
              <SectionItem
                key={s.key}
                id={`howto-${s.key}`}
                open={isOpen(s.key)}
                onToggle={() => toggle(s.key)}
                title={s.title}
              >
                {s.content}
              </SectionItem>
            ))}
          </div>

          {/* Footer */}
          <footer className="flex items-center justify-end gap-2 border-t border-base bg-card/80 px-4 py-2">
            <button
              onClick={onClose}
              className="rounded-xl border border-base bg-card px-4 py-2 text-sm shadow-sm transition hover:brightness-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
            >
              Zamknij
            </button>
            <button
              onClick={onClose}
              className="btn-primary rounded-xl px-4 py-2 text-sm font-medium shadow-sm transition hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
            >
              Zaczynam
            </button>
          </footer>
        </div>
      </section>

      {/* lokalne keyframes */}
      <style jsx>{`
        @keyframes auroraSlide {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </>
  );
}
