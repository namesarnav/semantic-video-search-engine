import * as React from "react";
import { cn } from "@/lib/utils";

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => (
  <input
    ref={ref}
    className={cn(
      "h-10 w-full rounded-lg border border-line bg-surface-2 px-3 text-sm",
      "text-fg placeholder:text-subtle transition-colors",
      "focus-visible:outline-none focus-visible:border-accent/60",
      "focus-visible:ring-2 focus-visible:ring-accent/25",
      "disabled:opacity-50",
      className,
    )}
    {...props}
  />
));
Input.displayName = "Input";
