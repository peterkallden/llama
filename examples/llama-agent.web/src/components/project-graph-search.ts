export function projectGraphSearchArguments(id: string, direction: "from" | "to", relation: string): Record<string, unknown> {
  const args: Record<string, unknown> = { [direction]: id };
  if (relation.length) args.relation = relation;
  return args;
}
