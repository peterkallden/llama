// TOAST UI 3.2.2's package exports omit its declarations and its published
// types reference private @t aliases. Describe the small pinned API we use
// here rather than disabling type checking for all dependencies.
declare module "@toast-ui/editor" {
  export default class Editor {
    constructor(options: {
      el: HTMLElement;
      height: string;
      initialValue: string;
      initialEditType: "markdown" | "wysiwyg";
      previewStyle: "vertical" | "tab";
      theme: "dark";
      usageStatistics: boolean;
      hideModeSwitch: boolean;
      toolbarItems: string[][];
    });
    getMarkdown(): string;
    on(event: "change", handler: () => void): void;
    destroy(): void;
  }
}

declare module "@toast-ui/editor/viewer" {
  export default class Viewer {
    constructor(options: { el: HTMLElement; initialValue: string; theme: "dark"; usageStatistics: boolean;
      linkAttributes: Record<string, string>; customHTMLSanitizer: (html: string) => string });
    setMarkdown(text: string): void;
    destroy(): void;
  }
}

declare module "dompurify" {
  const DOMPurify: { sanitize(html: string, options: { FORBID_TAGS: string[]; FORBID_ATTR: string[]; ALLOWED_URI_REGEXP: RegExp }): string };
  export default DOMPurify;
}
