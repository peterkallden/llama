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
