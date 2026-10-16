export type McpAppCspDomains = Record<
  "connectDomains" | "resourceDomains" | "frameDomains" | "baseUriDomains",
  string[]
>;

function normalizeHttpsDomain(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const match = value.trim().match(/^(?:https:\/\/)?(\*\.)?([a-z0-9.-]+)(?::([0-9]{1,5}))?$/i);
  if (!match) return undefined;
  const [, wildcard, hostname, rawPort] = match;
  const labels = hostname.split(".");
  if (labels.length < 2 || labels.some((label) =>
    !label || label.length > 63 || !/^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/i.test(label))) {
    return undefined;
  }
  if (rawPort && (Number(rawPort) < 1 || Number(rawPort) > 65535)) return undefined;
  return `https://${wildcard ?? ""}${hostname.toLowerCase()}${rawPort ? `:${Number(rawPort)}` : ""}`;
}

export function normalizeMcpAppCsp(meta: unknown): McpAppCspDomains {
  const outer = meta && typeof meta === "object" ? meta as Record<string, unknown> : {};
  const ui = outer.ui && typeof outer.ui === "object" ? outer.ui as Record<string, unknown> : outer;
  const csp = ui.csp && typeof ui.csp === "object" ? ui.csp as Record<string, unknown> : {};
  const domains = (field: keyof McpAppCspDomains) => Array.isArray(csp[field])
    ? [...new Set((csp[field] as unknown[]).map(normalizeHttpsDomain).filter((value): value is string => Boolean(value)))]
    : [];
  return {
    connectDomains: domains("connectDomains"),
    resourceDomains: domains("resourceDomains"),
    frameDomains: domains("frameDomains"),
    baseUriDomains: domains("baseUriDomains"),
  };
}

// Keep this as raw source because browsers parse the serialized script later;
// template-literal escape processing would corrupt its regular expressions.
export function sandboxProxyDocument(): string {
  return String.raw`<!doctype html><meta charset="utf-8"><script>
    let view;
    function sendHost(message){ parent.postMessage(message, '*'); }
    addEventListener('message', event => {
      if (event.source === parent && event.data?.method === 'ui/notifications/sandbox-resource-ready') {
        const { html, csp } = event.data.params || {};
        if (typeof html !== 'string' || !csp || typeof csp !== 'object') return;
        const domains = key => Array.isArray(csp[key])
          ? csp[key].filter(value => typeof value === 'string' && /^https:\/\/(?:\*\.)?[a-z0-9.-]+(?::[0-9]+)?$/i.test(value))
          : [];
        const resource = domains('resourceDomains').join(' ');
        const connect = domains('connectDomains').join(' ') || "'none'";
        const frames = domains('frameDomains').join(' ') || "'none'";
        const base = domains('baseUriDomains').join(' ') || "'none'";
        const policy = "default-src 'none'; script-src 'unsafe-inline' " + resource + "; style-src 'unsafe-inline' " + resource + "; img-src data: blob: " + resource + "; font-src " + resource + "; connect-src " + connect + "; frame-src " + frames + "; object-src 'none'; base-uri " + base + "; form-action 'none'";
        const meta = '<meta http-equiv="Content-Security-Policy" content="' + policy.replaceAll('&','&amp;').replaceAll('"','&quot;') + '">';
        const hasHead = /<head\b[^>]*>/i.test(html);
        const content = hasHead
          ? html.replace(/<head\b[^>]*>/i, head => head + meta)
          : '<!doctype html><html><head>' + meta + '</head><body>' + html + '</body></html>';
        view = document.createElement('iframe');
        view.setAttribute('sandbox', 'allow-scripts');
        view.setAttribute('referrerpolicy', 'no-referrer');
        view.style.cssText = 'border:0;width:100%;height:100%;min-height:560px';
        view.srcdoc = content;
        document.body.replaceChildren(view);
        return;
      }
      if (event.source === parent && view?.contentWindow && !String(event.data?.method || '').includes('/sandbox-')) {
        view.contentWindow.postMessage(event.data, '*');
      } else if (view?.contentWindow === event.source && event.origin === 'null' && !String(event.data?.method || '').includes('/sandbox-')) {
        sendHost(event.data);
      }
    });
    sendHost({jsonrpc:'2.0', method:'ui/notifications/sandbox-proxy-ready'});
  </scr${"ipt"}>`;
}

export type AppRequestGeneration = { generation: number; target: Window };

export function invocationNotifications(invocation: {
  input?: Record<string, unknown>; result?: unknown;
}): Record<string, unknown>[] {
  if (!invocation.input || Array.isArray(invocation.input)) return [];
  const messages: Record<string, unknown>[] = [{
    jsonrpc: "2.0", method: "ui/notifications/tool-input", params: { arguments: invocation.input },
  }];
  if (invocation.result && typeof invocation.result === "object" && !Array.isArray(invocation.result)) {
    const result = invocation.result as Record<string, unknown>;
    messages.push({
      jsonrpc: "2.0", method: "ui/notifications/tool-result",
      params: { ...result, content: Array.isArray(result.content) ? result.content : [] },
    });
  }
  return messages;
}

export function isCurrentAppRequest(
  request: AppRequestGeneration,
  generation: number,
  target: Window | null,
): boolean {
  return request.generation === generation && request.target === target;
}
