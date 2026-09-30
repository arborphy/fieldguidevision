import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const rows = JSON.parse(readFileSync(join(root, "analysis", "human_review_set.json"), "utf8"));
const quote = (value) => `'${String(value).replaceAll("'", "''")}'`;
const array = (values) => `array[${values.map(quote).join(",")}]::text[]`;
const batchId = "00000000-0000-4000-8000-000000000001";

const seedRows = rows.map((row) => `(
  '${batchId}', ${quote(row.image_id)}, ${row.review_index}, ${quote(row.selection_reason)}, ${array(row.error_modes || [])}
)`).join(",\n");

const sql = `-- Generated from analysis/human_review_set.json. Do not edit seed rows by hand.
create extension if not exists pgcrypto;

create table if not exists public.review_batches (
  id uuid primary key default gen_random_uuid(),
  slug text unique not null,
  title text not null,
  status text not null default 'open' check (status in ('open', 'closed')),
  created_at timestamptz not null default now()
);

create table if not exists public.review_items (
  batch_id uuid not null references public.review_batches(id) on delete cascade,
  image_id text not null,
  display_order integer not null,
  selection_reason text not null,
  error_modes text[] not null default '{}',
  primary key (batch_id, image_id),
  unique (batch_id, display_order)
);

create table if not exists public.annotations (
  batch_id uuid not null,
  image_id text not null,
  reviewer_id uuid not null default auth.uid() references auth.users(id),
  visible_tags text[] not null,
  primary_subject text,
  ambiguous boolean not null default false,
  other_tag text,
  note text,
  revision integer not null default 1,
  submitted_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (batch_id, image_id, reviewer_id),
  foreign key (batch_id, image_id) references public.review_items(batch_id, image_id) on delete cascade,
  check (cardinality(visible_tags) > 0),
  check (visible_tags <@ array['leaf','twig','bark','flower','fruit','cone','seed','whole plant','other']::text[])
);

create table if not exists public.annotation_events (
  id bigint generated always as identity primary key,
  batch_id uuid not null,
  image_id text not null,
  reviewer_id uuid not null default auth.uid() references auth.users(id),
  payload jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists annotations_reviewer_idx on public.annotations(reviewer_id, batch_id);
create index if not exists annotations_image_idx on public.annotations(batch_id, image_id);
create index if not exists annotation_events_reviewer_idx on public.annotation_events(reviewer_id, batch_id);

alter table public.review_batches enable row level security;
alter table public.review_items enable row level security;
alter table public.annotations enable row level security;
alter table public.annotation_events enable row level security;

drop policy if exists "authenticated reviewers read open batches" on public.review_batches;
create policy "authenticated reviewers read open batches" on public.review_batches for select to authenticated using (status = 'open');
drop policy if exists "authenticated reviewers read review items" on public.review_items;
create policy "authenticated reviewers read review items" on public.review_items for select to authenticated using (true);
drop policy if exists "reviewers read own annotations" on public.annotations;
create policy "reviewers read own annotations" on public.annotations for select to authenticated using (reviewer_id = auth.uid());
drop policy if exists "reviewers insert own annotations" on public.annotations;
create policy "reviewers insert own annotations" on public.annotations for insert to authenticated with check (reviewer_id = auth.uid());
drop policy if exists "reviewers update own annotations" on public.annotations;
create policy "reviewers update own annotations" on public.annotations for update to authenticated using (reviewer_id = auth.uid()) with check (reviewer_id = auth.uid());
drop policy if exists "reviewers read own events" on public.annotation_events;
create policy "reviewers read own events" on public.annotation_events for select to authenticated using (reviewer_id = auth.uid());
drop policy if exists "reviewers insert own events" on public.annotation_events;
create policy "reviewers insert own events" on public.annotation_events for insert to authenticated with check (reviewer_id = auth.uid());

grant select on public.review_batches, public.review_items to authenticated;
grant select, insert, update on public.annotations to authenticated;
grant select, insert on public.annotation_events to authenticated;
revoke delete on public.annotations, public.annotation_events from anon, authenticated;

create or replace function public.submit_annotation(
  p_batch_id uuid,
  p_image_id text,
  p_visible_tags text[],
  p_primary_subject text default null,
  p_ambiguous boolean default false,
  p_other_tag text default null,
  p_note text default null
) returns public.annotations
language plpgsql
security invoker
set search_path = public
as $$
declare
  saved public.annotations;
begin
  if auth.uid() is null then raise exception 'Authentication required'; end if;
  if cardinality(p_visible_tags) = 0 then raise exception 'Select at least one visible tag'; end if;
  insert into public.annotations (
    batch_id, image_id, reviewer_id, visible_tags, primary_subject, ambiguous, other_tag, note
  ) values (
    p_batch_id, p_image_id, auth.uid(), p_visible_tags, p_primary_subject, p_ambiguous, p_other_tag, p_note
  )
  on conflict (batch_id, image_id, reviewer_id) do update set
    visible_tags = excluded.visible_tags,
    primary_subject = excluded.primary_subject,
    ambiguous = excluded.ambiguous,
    other_tag = excluded.other_tag,
    note = excluded.note,
    revision = public.annotations.revision + 1,
    updated_at = now()
  returning * into saved;

  insert into public.annotation_events (batch_id, image_id, reviewer_id, payload)
  values (p_batch_id, p_image_id, auth.uid(), jsonb_build_object(
    'visible_tags', saved.visible_tags,
    'primary_subject', saved.primary_subject,
    'ambiguous', saved.ambiguous,
    'other_tag', saved.other_tag,
    'note', saved.note,
    'revision', saved.revision
  ));
  return saved;
end;
$$;

revoke all on function public.submit_annotation(uuid,text,text[],text,boolean,text,text) from public, anon;
grant execute on function public.submit_annotation(uuid,text,text[],text,boolean,text,text) to authenticated;

insert into public.review_batches (id, slug, title, status)
values ('${batchId}', 'targeted-100-v1', 'BioImages targeted Human Gold 100', 'open')
on conflict (id) do update set title = excluded.title;

insert into public.review_items (batch_id, image_id, display_order, selection_reason, error_modes)
values
${seedRows}
on conflict (batch_id, image_id) do update set
  display_order = excluded.display_order,
  selection_reason = excluded.selection_reason,
  error_modes = excluded.error_modes;
`;

const directory = join(root, "supabase", "migrations");
mkdirSync(directory, { recursive: true });
writeFileSync(join(directory, "001_human_review.sql"), sql);
console.log(`Wrote migration with ${rows.length} review items.`);
