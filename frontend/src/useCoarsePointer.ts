import { useEffect, useState } from "react";

export default function useCoarsePointer(): boolean {
  const [coarse, setCoarse] = useState(() => window.matchMedia("(pointer: coarse)").matches);

  useEffect(() => {
    const query = window.matchMedia("(pointer: coarse)");
    const update = () => setCoarse(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  return coarse;
}
