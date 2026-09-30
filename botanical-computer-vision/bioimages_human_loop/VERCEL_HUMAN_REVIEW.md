# Vercel Human Review deployment

## Recommendation

Deploy this site on **Vercel** and use the dedicated **BioImages Human Loop Supabase organization** for Postgres + Auth. It is deliberately separate from unrelated Supabase organizations and projects.

- Vercel serves the gallery and review UI.
- Supabase magic-link authentication gives each reviewer a stable identity without passwords.
- Supabase Postgres stores one annotation per reviewer and image.
- Row Level Security (RLS) lets reviewers read and update only their own answers.
- A server-only admin export can read every annotation and create consensus Human Gold.
- Images remain at BioImages; neither Vercel nor Supabase stores the 1,899 image files.

This is preferable to browser `localStorage`, which cannot synchronize across devices or reviewers. Vercel Blob and Edge Config are also the wrong primary store: the labels are structured, relational records that need constraints, revisions, exports, and agreement analysis.

Current Vercel documentation routes new relational databases through Marketplace integrations such as Supabase or Neon; the older first-party Vercel Postgres product is no longer offered. Supabase is the better fit here because it combines Postgres, authentication, and RLS in one integration.

## Reviewer experience

1. Reviewer opens the Vercel URL.
2. They enter an email and receive a magic link.
3. The app assigns the same deterministic 100-image batch, optionally in a reviewer-specific order.
4. Existing model/reference answers stay hidden until that reviewer submits the image.
5. The answer is saved immediately to Postgres.
6. The reviewer can resume from another device.
7. Admin views show completion and agreement; ordinary reviewers never see another person’s unfinished answers.

For a temporary public pilot, Supabase anonymous sign-in can replace magic links. It creates a real authenticated user ID without collecting PII, but the identity is lost if browser data is cleared. For research-quality Human Gold, magic links are more reliable.

## Database schema

```sql
create table public.review_batches (
  id uuid primary key default gen_random_uuid(),
  slug text unique not null,
  title text not null,
  status text not null default 'open' check (status in ('open', 'closed')),
  created_at timestamptz not null default now()
);

create table public.review_items (
  batch_id uuid not null references public.review_batches(id) on delete cascade,
  image_id text not null,
  display_order integer not null,
  selection_reason text not null,
  error_modes text[] not null default '{}',
  primary key (batch_id, image_id),
  unique (batch_id, display_order)
);

create table public.annotations (
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
  foreign key (batch_id, image_id)
    references public.review_items(batch_id, image_id) on delete cascade,
  check (cardinality(visible_tags) > 0),
  check (visible_tags <@ array[
    'leaf','twig','bark','flower','fruit','cone','seed','whole plant','other'
  ]::text[])
);

create table public.annotation_events (
  id bigint generated always as identity primary key,
  batch_id uuid not null,
  image_id text not null,
  reviewer_id uuid not null references auth.users(id),
  payload jsonb not null,
  created_at timestamptz not null default now()
);

create index annotations_reviewer_idx
  on public.annotations(reviewer_id, batch_id);
create index annotations_image_idx
  on public.annotations(batch_id, image_id);
```

`annotation_events` is append-only audit history. `annotations` contains the latest answer used by the UI and analysis.

## Access policy

Enable RLS on every exposed table. Reviewers may read the batch and item list, but may only select, insert, and update annotations whose `reviewer_id = auth.uid()`. Do not grant reviewer deletes; a correction is a new revision and audit event. The Supabase secret/service key must exist only in a Vercel server environment variable because it bypasses RLS.

The public website should never expose:

- Supabase secret/service key
- database connection string
- other reviewers’ raw labels
- aggregate Human Gold before the reviewer submits the current image

## Consensus and export

Do not overwrite individual annotations with a single shared value. Keep every reviewer response, then produce a derived consensus table or CSV:

- per-tag vote count and agreement;
- majority/threshold consensus;
- adjudication state for disagreements;
- reviewer count;
- final accepted tag set.

This preserves inter-annotator agreement and lets the project distinguish genuine ambiguity from labeling mistakes.

## Implemented deployment files

- `vercel.json`: builds the static site into `dist/`.
- `scripts/build_vercel.mjs`: injects the public Supabase URL/key from Vercel environment variables without committing them to Git.
- `data/runtime-config.js`: empty safe fallback used by GitHub Pages.
- `supabase/migrations/001_human_review.sql`: schema, RLS policies, transactional submit function, audit events, and the 100-item seed batch.
- `assets/app.js`: magic-link sign-in, per-reviewer synchronization, revision-safe saves, local preview fallback, and JSON/CSV export.

## Required external setup

The remaining external setup requires access to the project owner’s accounts:

1. create the database in the dedicated BioImages Human Loop Supabase organization;
2. apply `supabase/migrations/001_human_review.sql`;
3. configure the final Vercel URL as an allowed Auth redirect;
4. set `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and `REVIEW_BATCH_ID` in Vercel;
5. deploy and test two independent reviewer accounts.

Official references:

- <https://vercel.com/docs/storage>
- <https://vercel.com/marketplace/supabase/supabase>
- <https://supabase.com/docs/guides/auth>
- <https://supabase.com/docs/guides/database/postgres/row-level-security>
