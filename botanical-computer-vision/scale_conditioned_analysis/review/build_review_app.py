#!/usr/bin/env python3
"""Build the browser payload and Supabase migration for scale review."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_CV = ROOT.parent
REVIEW_ROOT = Path(__file__).resolve().parent
DATA_DIR = REVIEW_ROOT / "data"
MIGRATION = REPO_CV / "bioimages_human_loop" / "supabase" / "migrations" / "002_scale_review.sql"
BATCH_ID = "00000000-0000-4000-8000-000000000002"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sql_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> None:
    review_rows = read_csv(ROOT / "data" / "scale_review_sample.csv")
    manifest = {
        row["id"]: row
        for row in read_csv(REPO_CV / "bioimages_human_loop" / "data" / "image_manifest.csv")
    }
    codebook = json.loads((ROOT / "data" / "scale_codebook.json").read_text(encoding="utf-8"))

    examples: dict[str, list[dict[str, str]]] = {}
    for label, rows in codebook.items():
        examples[label] = []
        for row in rows[:3]:
            source = manifest.get(row["image_id"], {})
            examples[label].append(
                {
                    "image_id": row["image_id"],
                    "species": row["species"],
                    "caption": f'{row["organ_category"]} · {row["subview"]}',
                    "image_url": source.get("thumbnail_url") or source.get("image_url") or row["image_url"],
                    "source_url": row["source_url"],
                }
            )

    items = []
    for index, row in enumerate(review_rows, start=1):
        items.append(
            {
                "index": index,
                "image_id": row["image_id"],
                "species": row["species"],
                "organ_category": row["organ_category"],
                "subview": row["subview"],
                "image_url": row["image_url"],
                "source_url": row["source_url"],
                "provisional_scale": row["shot_scale"],
                "provisional_confidence": row["shot_scale_confidence"],
                "rationale": row["shot_scale_rationale"],
                "evidence": {
                    "distant": [row["source_distant_evidence"], row["gemma_distant_evidence"]],
                    "mid-range": [row["source_mid_evidence"], row["gemma_mid_evidence"]],
                    "close-up": [row["source_close_evidence"], row["gemma_close_evidence"]],
                },
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": BATCH_ID,
        "labels": ["distant", "mid-range", "close-up", "uncertain"],
        "definitions": {
            "distant": "The whole plant, tree, shrub, or most of the crown is visible; overall habit and scene context are readable.",
            "mid-range": "A branch system, trunk section, or several connected organs are visible; the whole organism is outside the frame.",
            "close-up": "One organ, surface, or fine structure dominates the frame; local texture or shape is readable.",
            "uncertain": "More than one rule fits, the frame sits on a boundary, or the visible evidence is insufficient.",
        },
        "examples": examples,
        "items": items,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    (DATA_DIR / "scale-review-data.js").write_text(
        f"window.BIOIMAGES_SCALE_REVIEW = {serialized};\n", encoding="utf-8"
    )

    seed_values = []
    for item in items:
        seed_values.append(
            "(" + ", ".join(
                [
                    sql_text(BATCH_ID),
                    sql_text(item["image_id"]),
                    str(item["index"]),
                    sql_text(item["species"]),
                    sql_text(item["organ_category"]),
                    sql_text(item["subview"]),
                    sql_text(item["provisional_scale"]),
                    sql_text(item["provisional_confidence"]),
                ]
            ) + ")"
        )

    migration = f"""-- Shared anonymous review for the 92-image photographic-scale audit.
-- Anonymous Supabase users still receive auth.uid() and the authenticated role.

create table if not exists public.scale_review_batches (
  id uuid primary key,
  slug text unique not null,
  title text not null,
  status text not null default 'open' check (status in ('open', 'closed')),
  created_at timestamptz not null default now()
);

create table if not exists public.scale_review_items (
  batch_id uuid not null references public.scale_review_batches(id) on delete cascade,
  image_id text not null,
  display_order integer not null,
  species text not null,
  organ_category text not null,
  subview text not null,
  provisional_scale text not null,
  provisional_confidence text not null,
  primary key (batch_id, image_id),
  unique (batch_id, display_order)
);

create table if not exists public.scale_annotations (
  batch_id uuid not null,
  image_id text not null,
  reviewer_id uuid not null default auth.uid() references auth.users(id),
  human_scale text not null check (human_scale in ('distant', 'mid-range', 'close-up', 'uncertain')),
  confidence text not null check (confidence in ('high', 'medium', 'low')),
  note text,
  revision integer not null default 1,
  submitted_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (batch_id, image_id, reviewer_id),
  foreign key (batch_id, image_id)
    references public.scale_review_items(batch_id, image_id) on delete cascade
);

create table if not exists public.scale_annotation_events (
  id bigint generated always as identity primary key,
  batch_id uuid not null,
  image_id text not null,
  reviewer_id uuid not null references auth.users(id),
  payload jsonb not null,
  created_at timestamptz not null default now()
);

alter table public.scale_review_batches enable row level security;
alter table public.scale_review_items enable row level security;
alter table public.scale_annotations enable row level security;
alter table public.scale_annotation_events enable row level security;

drop policy if exists "scale batches readable" on public.scale_review_batches;
create policy "scale batches readable" on public.scale_review_batches
  for select to authenticated using (true);

drop policy if exists "scale items readable" on public.scale_review_items;
create policy "scale items readable" on public.scale_review_items
  for select to authenticated using (true);

drop policy if exists "reviewers read own scale annotations" on public.scale_annotations;
create policy "reviewers read own scale annotations" on public.scale_annotations
  for select to authenticated using (reviewer_id = auth.uid());

drop policy if exists "reviewers insert own scale annotations" on public.scale_annotations;
create policy "reviewers insert own scale annotations" on public.scale_annotations
  for insert to authenticated with check (reviewer_id = auth.uid());

drop policy if exists "reviewers update own scale annotations" on public.scale_annotations;
create policy "reviewers update own scale annotations" on public.scale_annotations
  for update to authenticated using (reviewer_id = auth.uid()) with check (reviewer_id = auth.uid());

drop policy if exists "reviewers read own scale events" on public.scale_annotation_events;
create policy "reviewers read own scale events" on public.scale_annotation_events
  for select to authenticated using (reviewer_id = auth.uid());

drop policy if exists "reviewers insert own scale events" on public.scale_annotation_events;
create policy "reviewers insert own scale events" on public.scale_annotation_events
  for insert to authenticated with check (reviewer_id = auth.uid());

grant select on public.scale_review_batches, public.scale_review_items to authenticated;
grant select, insert, update on public.scale_annotations to authenticated;
grant select, insert on public.scale_annotation_events to authenticated;
revoke delete on public.scale_annotations, public.scale_annotation_events from anon, authenticated;

create or replace function public.submit_scale_annotation(
  p_batch_id uuid,
  p_image_id text,
  p_human_scale text,
  p_confidence text,
  p_note text default null
) returns public.scale_annotations
language plpgsql
security invoker
set search_path = public
as $$
declare
  result public.scale_annotations;
begin
  if auth.uid() is null then
    raise exception 'Authentication required';
  end if;

  insert into public.scale_annotations (
    batch_id, image_id, reviewer_id, human_scale, confidence, note
  ) values (
    p_batch_id, p_image_id, auth.uid(), p_human_scale, p_confidence, nullif(trim(p_note), '')
  )
  on conflict (batch_id, image_id, reviewer_id) do update set
    human_scale = excluded.human_scale,
    confidence = excluded.confidence,
    note = excluded.note,
    revision = public.scale_annotations.revision + 1,
    updated_at = now()
  returning * into result;

  insert into public.scale_annotation_events (batch_id, image_id, reviewer_id, payload)
  values (p_batch_id, p_image_id, auth.uid(), to_jsonb(result));

  return result;
end;
$$;

revoke all on function public.submit_scale_annotation(uuid, text, text, text, text) from public, anon;
grant execute on function public.submit_scale_annotation(uuid, text, text, text, text) to authenticated;

insert into public.scale_review_batches (id, slug, title)
values ('{BATCH_ID}', 'scale-audit-v1', 'BioImages photographic-scale audit')
on conflict (id) do update set title = excluded.title;

insert into public.scale_review_items (
  batch_id, image_id, display_order, species, organ_category, subview,
  provisional_scale, provisional_confidence
) values
{',\n'.join(seed_values)}
on conflict (batch_id, image_id) do update set
  display_order = excluded.display_order,
  species = excluded.species,
  organ_category = excluded.organ_category,
  subview = excluded.subview,
  provisional_scale = excluded.provisional_scale,
  provisional_confidence = excluded.provisional_confidence;
"""
    MIGRATION.write_text(migration, encoding="utf-8")
    print(f"Wrote {len(items)} review items to {DATA_DIR / 'scale-review-data.js'}")
    print(f"Wrote migration to {MIGRATION}")


if __name__ == "__main__":
    main()
