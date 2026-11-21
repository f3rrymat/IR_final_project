import clsx from "clsx";
import { Info, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import { clamp01, methodKind, methodLabel, type MethodKind } from "../lib/format";

/** Segmented control: page tabs, collection switcher, method picker, split picker. */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  size = "md",
  className,
}: {
  value: T;
  options: { value: T; label: ReactNode; disabled?: boolean; title?: string }[];
  onChange: (v: T) => void;
  label: string;
  size?: "sm" | "md";
  className?: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={clsx("inline-flex max-w-full flex-wrap gap-0.5 rounded-ctl bg-sunken p-[3px]", className)}
    >
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={on}
            disabled={o.disabled}
            title={o.title}
            onClick={() => onChange(o.value)}
            className={clsx(
              "inline-flex items-center gap-1.5 whitespace-nowrap rounded-[7px] font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40",
              size === "md" ? "px-3 py-1.5 text-[0.86rem]" : "px-2.5 py-1 text-[0.8rem]",
              on ? "bg-surface text-ink shadow-panel" : "text-ink-soft hover:text-ink",
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export function Chip({
  children,
  active,
  onClick,
  title,
}: {
  children: ReactNode;
  active?: boolean;
  onClick?: () => void;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-pressed={active}
      className={clsx(
        "rounded-ctl border px-2.5 py-1 text-left text-[0.82rem] transition-colors duration-150",
        active
          ? "border-accent/40 bg-accent-soft text-accent-ink"
          : "border-line/[0.12] bg-surface text-ink-soft hover:border-line/25 hover:text-ink",
      )}
    >
      {children}
    </button>
  );
}

type Tone = "plain" | "accent" | "pass" | "fail" | "semantic" | "expand";
const TONE: Record<Tone, string> = {
  plain: "bg-sunken text-ink-soft",
  accent: "bg-accent-soft text-accent-ink",
  pass: "bg-pass-soft text-pass",
  fail: "bg-fail-soft text-fail",
  semantic: "bg-semantic-soft text-semantic",
  expand: "bg-expand-soft text-expand",
};

export function Badge({ children, tone = "plain", className }: { children: ReactNode; tone?: Tone; className?: string }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-[6px] px-1.5 py-px text-[0.74rem] font-medium",
        TONE[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

const KIND_DOT: Record<MethodKind, string> = {
  lexical: "bg-lexical",
  semantic: "bg-semantic",
  expand: "bg-expand",
  fused: "bg-ink",
};

/** A method name with its family colour dot. */
export function MethodTag({ method, className }: { method: string; className?: string }) {
  return (
    <span className={clsx("inline-flex items-center gap-1.5 whitespace-nowrap", className)}>
      <span className={clsx("h-2 w-2 flex-none rounded-full", KIND_DOT[methodKind(method)])} aria-hidden />
      {methodLabel(method)}
    </span>
  );
}

export const kindBg = (m: string) => KIND_DOT[methodKind(m)];

/** Horizontal meter; `marker` draws a threshold tick (e.g. tau). */
export function Meter({
  value,
  marker,
  tone = "accent",
  className,
  label,
}: {
  value: number;
  marker?: number | null;
  tone?: "accent" | "pass" | "fail" | "mute" | "semantic" | "lexical";
  className?: string;
  label?: string;
}) {
  const v = clamp01(value);
  return (
    <div
      className={clsx("relative h-1.5 rounded-full bg-sunken", className)}
      role="meter"
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuenow={Number(v.toFixed(3))}
      aria-label={label}
    >
      <span
        className={clsx(
          "block h-full rounded-full transition-[width] duration-500 ease-out",
          tone === "accent" && "bg-accent",
          tone === "lexical" && "bg-lexical",
          tone === "semantic" && "bg-semantic",
          tone === "pass" && "bg-pass",
          tone === "fail" && "bg-fail",
          tone === "mute" && "bg-ink-faint/60",
        )}
        style={{ width: `${v * 100}%` }}
      />
      {marker != null && (
        <span
          className="absolute -top-[3px] h-[calc(100%+6px)] w-[2px] rounded-full bg-ink"
          style={{ left: `calc(${clamp01(marker) * 100}% - 1px)` }}
          aria-hidden
        />
      )}
    </div>
  );
}

/** A labelled figure. The label and value share a parent so screen readers and tests read them together. */
export function Stat({
  label,
  value,
  sub,
  note,
  tone,
  children,
  className,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  note?: ReactNode;
  tone?: "pass" | "fail";
  children?: ReactNode;
  className?: string;
}) {
  return (
    <div className={clsx("min-w-0", className)}>
      <div className="label">{label}</div>
      <div
        className={clsx(
          "num mt-0.5 text-[1.6rem] font-semibold leading-tight tracking-[-0.02em]",
          tone === "pass" && "text-pass",
          tone === "fail" && "text-fail",
        )}
      >
        {value} {sub && <span className="text-[0.82rem] font-normal tracking-normal text-ink-faint">{sub}</span>}
      </div>
      {note && <div className="mt-0.5 text-[0.8rem] leading-snug text-ink-soft">{note}</div>}
      {children && <div className="mt-2">{children}</div>}
    </div>
  );
}

/** A titled section of a page. */
export function Panel({
  title,
  aside,
  lead,
  children,
  id,
  className,
}: {
  title: string;
  aside?: ReactNode;
  lead?: ReactNode;
  children?: ReactNode;
  id?: string;
  className?: string;
}) {
  return (
    <section className={clsx("mt-12", className)} id={id} aria-label={title}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 className="text-[1.15rem] font-semibold tracking-[-0.01em]">{title}</h2>
        {aside}
      </div>
      {lead && <p className="mt-1.5 max-w-[72ch] text-[0.9rem] leading-relaxed text-ink-soft">{lead}</p>}
      {children}
    </section>
  );
}

export function Callout({
  tone = "plain",
  children,
  title,
  className,
}: {
  tone?: "note" | "warn" | "plain" | "info";
  children: ReactNode;
  title?: string;
  className?: string;
}) {
  const Icon = tone === "warn" ? TriangleAlert : Info;
  return (
    <div
      role={tone === "warn" ? "alert" : "note"}
      className={clsx(
        "mt-4 flex gap-3 rounded-ctl border px-4 py-3 text-[0.9rem] leading-relaxed",
        tone === "note" && "border-pass/20 bg-pass-soft text-ink",
        tone === "warn" && "border-fail/25 bg-fail-soft text-ink",
        tone === "plain" && "border-line/[0.08] bg-raised text-ink-soft",
        tone === "info" && "border-accent/20 bg-accent-soft text-ink",
        className,
      )}
    >
      <Icon
        className={clsx("mt-[3px] h-4 w-4 flex-none", tone === "warn" ? "text-fail" : tone === "note" ? "text-pass" : "text-accent")}
        aria-hidden
      />
      <div className="min-w-0">
        {title && <b className="font-semibold text-ink">{title} </b>}
        {children}
      </div>
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
  disabled,
  disabledHint,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
  hint?: ReactNode;
  disabled?: boolean;
  disabledHint?: string;
}) {
  return (
    <label
      className={clsx("flex cursor-pointer items-start gap-3 py-1.5", disabled && "cursor-not-allowed opacity-50")}
      title={disabled ? disabledHint : undefined}
    >
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={clsx(
          "relative mt-0.5 h-[18px] w-8 flex-none rounded-full transition-colors duration-200",
          checked ? "bg-accent" : "bg-ink-faint/35",
        )}
      >
        <span
          className={clsx(
            "absolute top-[2px] h-[14px] w-[14px] rounded-full bg-white shadow transition-transform duration-200 ease-out",
            checked ? "translate-x-[16px]" : "translate-x-[2px]",
          )}
        />
      </button>
      <span className="text-[0.88rem]">
        <span className="font-medium text-ink">{label}</span>
        {hint && <span className="block text-[0.8rem] leading-snug text-ink-soft">{hint}</span>}
        {disabled && disabledHint && <span className="block text-[0.76rem] text-ink-faint">{disabledHint}</span>}
      </span>
    </label>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("skeleton", className)} aria-hidden />;
}

export function TableWrap({ children, label, className }: { children: ReactNode; label: string; className?: string }) {
  return (
    <div className={clsx("panel mt-4 overflow-x-auto", className)} role="region" aria-label={label} tabIndex={0}>
      <table className="w-full border-collapse text-[0.88rem]">{children}</table>
    </div>
  );
}

export const th =
  "whitespace-nowrap border-b hairline bg-raised px-4 py-2.5 text-left text-[0.78rem] font-medium text-ink-faint first:pl-5";
export const thNum = th + " text-right";
export const td = "border-b hairline px-4 py-2.5 align-middle first:pl-5";
export const tdNum = td + " num text-right tabular-nums";
