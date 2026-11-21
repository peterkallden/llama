// A request remains current only within its captured scope and operation.
// Independent views use separate gates so refreshing one does not cancel another.
export function createLatestRequestGate(readScope: () => unknown) {
  let sequence = 0;
  return () => {
    const request = ++sequence;
    const scope = readScope();
    return () => request === sequence && Object.is(scope, readScope());
  };
}
