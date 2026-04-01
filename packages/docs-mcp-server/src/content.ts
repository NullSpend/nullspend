import docsData from "./generated/docs.json" with { type: "json" };

export interface DocPage {
  path: string;
  title: string;
  description: string;
  content: string;
}

export const DOCS: DocPage[] = docsData as DocPage[];

/** Map from normalized path → DocPage for O(1) lookup */
export const DOCS_BY_PATH: ReadonlyMap<string, DocPage> = new Map(
  DOCS.map((doc) => [doc.path, doc]),
);
