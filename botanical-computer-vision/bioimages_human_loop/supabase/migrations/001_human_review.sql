-- Generated from analysis/human_review_set.json. Do not edit seed rows by hand.
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
values ('00000000-0000-4000-8000-000000000001', 'targeted-100-v1', 'BioImages targeted Human Gold 100', 'open')
on conflict (id) do update set title = excluded.title;

insert into public.review_items (batch_id, image_id, display_order, selection_reason, error_modes)
values
(
  '00000000-0000-4000-8000-000000000001', 'thomas/0506-01-06', 1, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers fruit state view confusion; covers high confidence extra tag; covers leaf view confusion; covers reproductive structure missed; covers secondary woody structure missed; covers whole plant scale split; model disagreement; covers tag fruit; covers tag leaf; covers tag twig; covers tag whole plant', array['bioimages_single_label_ambiguity','flower_fruit_bud_ambiguity','fruit_state_view_confusion','high_confidence_extra_tag','leaf_dominates_reproductive','leaf_view_confusion','reproductive_structure_missed','secondary_woody_structure_missed','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/50588', 2, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers fruit state view confusion; covers high confidence extra tag; covers leaf view confusion; covers reproductive structure missed; covers strong model disagreement; model disagreement; rare class; covers tag cone; covers tag fruit; covers tag twig', array['bioimages_single_label_ambiguity','cone_seed_under_detection','flower_fruit_bud_ambiguity','fruit_state_view_confusion','high_confidence_extra_tag','leaf_view_confusion','reproductive_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/28649', 3, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers high confidence extra tag; covers secondary woody structure missed; covers strong model disagreement; covers whole plant scale split; model disagreement; covers tag flower; covers tag fruit; covers tag leaf; covers tag twig; covers tag whole plant', array['bioimages_single_label_ambiguity','flower_fruit_bud_ambiguity','high_confidence_extra_tag','leaf_dominates_reproductive','secondary_woody_structure_missed','strong_model_disagreement','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/13534', 4, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers fruit state view confusion; covers high confidence extra tag; covers leaf view confusion; covers reproductive structure missed; covers secondary woody structure missed; covers whole plant scale split; model disagreement; covers tag leaf; covers tag twig; covers tag whole plant', array['flower_fruit_bud_ambiguity','fruit_state_view_confusion','high_confidence_extra_tag','leaf_dominates_reproductive','leaf_view_confusion','reproductive_structure_missed','secondary_woody_structure_missed','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/24001', 5, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers high confidence extra tag; covers reproductive structure missed; covers strong model disagreement; model disagreement; rare class; covers tag fruit; covers tag leaf; covers tag seed; covers tag twig', array['cone_seed_under_detection','flower_fruit_bud_ambiguity','high_confidence_extra_tag','reproductive_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/11054', 6, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers high confidence extra tag; covers reproductive structure missed; covers whole plant scale split; model disagreement; rare class; covers tag fruit; covers tag leaf; covers tag seed; covers tag whole plant', array['cone_seed_under_detection','flower_fruit_bud_ambiguity','high_confidence_extra_tag','reproductive_structure_missed','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/66156', 7, 'gemma disagreement; covers flower fruit bud ambiguity; covers fruit state view confusion; covers leaf view confusion; covers reproductive structure missed; covers secondary woody structure missed; model disagreement; covers tag bark; covers tag flower; covers tag fruit; covers tag leaf; covers tag twig', array['flower_fruit_bud_ambiguity','fruit_state_view_confusion','leaf_view_confusion','reproductive_structure_missed','secondary_woody_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'thomas/0014-01-06', 8, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers fruit state view confusion; covers high confidence extra tag; covers reproductive structure missed; covers strong model disagreement; covers whole plant scale split; model disagreement; covers tag flower; covers tag leaf; covers tag whole plant', array['bioimages_single_label_ambiguity','flower_fruit_bud_ambiguity','fruit_state_view_confusion','high_confidence_extra_tag','leaf_dominates_reproductive','reproductive_structure_missed','strong_model_disagreement','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/38128', 9, 'bioimages mismatch; gemma disagreement; high confidence error; covers high confidence extra tag; covers leaf view confusion; covers strong model disagreement; model disagreement; rare class; covers tag cone; covers tag twig', array['cone_seed_under_detection','high_confidence_extra_tag','leaf_view_confusion','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/54680', 10, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers fruit state view confusion; covers high confidence extra tag; covers secondary woody structure missed; model disagreement; covers tag bark; covers tag fruit; covers tag leaf; covers tag twig', array['flower_fruit_bud_ambiguity','fruit_state_view_confusion','high_confidence_extra_tag','secondary_woody_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/11091', 11, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers high confidence extra tag; covers reproductive structure missed; covers strong model disagreement; model disagreement; rare class; covers tag cone; covers tag flower', array['cone_seed_under_detection','flower_fruit_bud_ambiguity','high_confidence_extra_tag','reproductive_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/24849', 12, 'bioimages mismatch; gemma disagreement; high confidence error; covers fruit state view confusion; covers high confidence extra tag; covers reproductive structure missed; covers strong model disagreement; model disagreement; rare class; covers tag fruit; covers tag seed', array['cone_seed_under_detection','fruit_state_view_confusion','high_confidence_extra_tag','reproductive_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/49296', 13, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers high confidence extra tag; covers reproductive structure missed; covers secondary woody structure missed; covers whole plant scale split; model disagreement; covers tag flower; covers tag leaf; covers tag whole plant', array['bioimages_single_label_ambiguity','flower_fruit_bud_ambiguity','high_confidence_extra_tag','leaf_dominates_reproductive','reproductive_structure_missed','secondary_woody_structure_missed','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/50384', 14, 'bioimages mismatch; gemma disagreement; covers leaf view confusion; covers secondary woody structure missed; covers strong model disagreement; model disagreement; rare class; covers tag bark; covers tag twig', array['leaf_view_confusion','secondary_woody_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/55230', 15, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers fruit state view confusion; covers high confidence extra tag; covers reproductive structure missed; model disagreement; rare class; covers tag fruit; covers tag seed', array['cone_seed_under_detection','flower_fruit_bud_ambiguity','fruit_state_view_confusion','high_confidence_extra_tag','reproductive_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/13122', 16, 'easy case; high confidence error; covers high confidence extra tag; covers whole plant scale split; covers tag bark; covers tag leaf; covers tag whole plant', array['bark_scale_confusion','high_confidence_extra_tag','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/50745', 17, 'bioimages mismatch; covers flower fruit bud ambiguity; covers reproductive structure missed; covers strong model disagreement; model disagreement; covers tag bark; covers tag cone; covers tag flower; covers tag twig', array['bioimages_single_label_ambiguity','cone_seed_under_detection','flower_fruit_bud_ambiguity','reproductive_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/50556', 18, 'easy case; rare class; covers tag bark; covers tag whole plant', array['bark_scale_confusion']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/50503', 19, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers fruit state view confusion; covers high confidence extra tag; covers reproductive structure missed; covers strong model disagreement; model disagreement; rare class; covers tag cone; covers tag fruit', array['cone_seed_under_detection','flower_fruit_bud_ambiguity','fruit_state_view_confusion','high_confidence_extra_tag','reproductive_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/10805', 20, 'bioimages mismatch; gemma disagreement; covers flower fruit bud ambiguity; covers fruit state view confusion; covers leaf view confusion; covers strong model disagreement; model disagreement; rare class; covers tag cone; covers tag twig', array['cone_seed_under_detection','flower_fruit_bud_ambiguity','fruit_state_view_confusion','leaf_view_confusion','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/38154', 21, 'easy case; covers whole plant scale split; covers tag bark; covers tag leaf; covers tag whole plant', array['bark_scale_confusion','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/20950', 22, 'bioimages mismatch; gemma disagreement; covers flower fruit bud ambiguity; covers fruit state view confusion; covers reproductive structure missed; covers strong model disagreement; model disagreement; rare class; covers tag fruit; covers tag seed', array['cone_seed_under_detection','flower_fruit_bud_ambiguity','fruit_state_view_confusion','reproductive_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/10497', 23, 'easy case; high confidence error; covers high confidence extra tag; covers tag flower; covers tag twig; covers tag whole plant', array['high_confidence_extra_tag']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/50977', 24, 'easy case; covers flower fruit bud ambiguity; covers tag flower', array['flower_fruit_bud_ambiguity']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/39501', 25, 'bioimages mismatch; gemma disagreement; covers flower fruit bud ambiguity; covers fruit state view confusion; covers reproductive structure missed; covers strong model disagreement; model disagreement; rare class; covers tag fruit; covers tag seed', array['cone_seed_under_detection','flower_fruit_bud_ambiguity','fruit_state_view_confusion','leaf_dominates_reproductive','reproductive_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/12361', 26, 'easy case; high confidence error; covers high confidence extra tag; covers whole plant scale split; covers tag bark; covers tag leaf; covers tag whole plant', array['bark_scale_confusion','high_confidence_extra_tag','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/38294', 27, 'easy case; rare class; covers tag whole plant', array[]::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'ssmv/2-886-03', 28, 'easy case; high confidence error; covers high confidence extra tag; covers tag bark; covers tag whole plant', array['bark_scale_confusion','high_confidence_extra_tag']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/43578', 29, 'bioimages mismatch; gemma disagreement; high confidence error; covers fruit state view confusion; covers high confidence extra tag; covers reproductive structure missed; rare class; covers tag seed', array['cone_seed_under_detection','fruit_state_view_confusion','high_confidence_extra_tag','reproductive_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/54972', 30, 'easy case; covers leaf view confusion; covers tag leaf', array['leaf_view_confusion']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/11019', 31, 'easy case; covers leaf view confusion; covers tag leaf', array['leaf_view_confusion']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/18969', 32, 'high confidence error; covers flower fruit bud ambiguity; covers high confidence extra tag; covers secondary woody structure missed; model disagreement; covers tag flower', array['flower_fruit_bud_ambiguity','high_confidence_extra_tag','secondary_woody_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/13075', 33, 'easy case; covers tag leaf', array[]::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/30278', 34, 'covers fruit state view confusion; model disagreement; rare class; covers tag fruit; covers tag seed', array['cone_seed_under_detection','fruit_state_view_confusion']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/51409', 35, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers high confidence extra tag; covers leaf view confusion; covers reproductive structure missed; covers strong model disagreement; model disagreement; covers tag flower; covers tag leaf', array['flower_fruit_bud_ambiguity','high_confidence_extra_tag','leaf_dominates_reproductive','leaf_view_confusion','reproductive_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'ssmv/2-253-02', 36, 'easy case; high confidence error; covers high confidence extra tag; covers whole plant scale split; covers tag leaf; covers tag whole plant', array['high_confidence_extra_tag','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/30435', 37, 'bioimages mismatch; gemma disagreement; covers flower fruit bud ambiguity; covers fruit state view confusion; covers reproductive structure missed; covers strong model disagreement; model disagreement; covers tag fruit; covers tag twig', array['flower_fruit_bud_ambiguity','fruit_state_view_confusion','reproductive_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/11996', 38, 'bioimages mismatch; high confidence error; covers high confidence extra tag; covers leaf view confusion; covers whole plant scale split; model disagreement; covers tag leaf; covers tag whole plant', array['high_confidence_extra_tag','leaf_view_confusion','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/89370', 39, 'high confidence error; covers high confidence extra tag; covers secondary woody structure missed; covers whole plant scale split; model disagreement; covers tag twig; covers tag whole plant', array['high_confidence_extra_tag','secondary_woody_structure_missed','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/39152', 40, 'gemma disagreement; high confidence error; covers high confidence extra tag; model disagreement; covers tag twig', array['high_confidence_extra_tag']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/30763', 41, 'bioimages mismatch; covers flower fruit bud ambiguity; covers reproductive structure missed; covers whole plant scale split; model disagreement; covers tag fruit; covers tag twig; covers tag whole plant', array['bioimages_single_label_ambiguity','flower_fruit_bud_ambiguity','reproductive_structure_missed','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'kirchoff/em2319', 42, 'covers flower fruit bud ambiguity; covers secondary woody structure missed; model disagreement; covers tag leaf; covers tag twig', array['flower_fruit_bud_ambiguity','secondary_woody_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/50630', 43, 'bioimages mismatch; gemma disagreement; covers flower fruit bud ambiguity; covers fruit state view confusion; covers reproductive structure missed; covers strong model disagreement; model disagreement; covers tag flower; covers tag fruit; covers tag twig', array['flower_fruit_bud_ambiguity','fruit_state_view_confusion','reproductive_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/15176', 44, 'covers leaf view confusion; covers tag leaf', array['leaf_view_confusion']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/12405', 45, 'high confidence error; covers flower fruit bud ambiguity; covers fruit state view confusion; covers high confidence extra tag; model disagreement; covers tag fruit; covers tag leaf', array['flower_fruit_bud_ambiguity','fruit_state_view_confusion','high_confidence_extra_tag']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/10904', 46, 'bioimages mismatch; covers whole plant scale split; model disagreement; covers tag leaf; covers tag whole plant', array['bioimages_single_label_ambiguity','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/35422', 47, 'high confidence error; covers high confidence extra tag; model disagreement; covers tag bark', array['bark_scale_confusion','high_confidence_extra_tag']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/35184', 48, 'bioimages mismatch; gemma disagreement; covers leaf view confusion; covers secondary woody structure missed; covers strong model disagreement; model disagreement; covers tag leaf', array['leaf_view_confusion','secondary_woody_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'kirchoff/b5070', 49, 'model disagreement; covers tag bark', array[]::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'kirchoff/em2062', 50, 'easy case; high confidence error; covers high confidence extra tag; covers leaf view confusion; covers tag leaf', array['high_confidence_extra_tag','leaf_view_confusion']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/26528', 51, 'high confidence error; covers high confidence extra tag; covers strong model disagreement; model disagreement; covers tag bark; covers tag twig', array['bark_scale_confusion','high_confidence_extra_tag','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/65649', 52, 'high confidence error; covers high confidence extra tag; covers leaf view confusion; covers secondary woody structure missed; covers strong model disagreement; model disagreement; covers tag leaf; covers tag twig', array['high_confidence_extra_tag','leaf_view_confusion','secondary_woody_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/90899', 53, 'high confidence error; covers high confidence extra tag; covers whole plant scale split; model disagreement; covers tag leaf; covers tag whole plant', array['high_confidence_extra_tag','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/18855', 54, 'high confidence error; covers flower fruit bud ambiguity; covers fruit state view confusion; covers high confidence extra tag; covers secondary woody structure missed; model disagreement; covers tag flower; covers tag fruit; covers tag twig', array['flower_fruit_bud_ambiguity','fruit_state_view_confusion','high_confidence_extra_tag','secondary_woody_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/10534', 55, 'model disagreement; covers tag flower', array[]::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/10492', 56, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers fruit state view confusion; covers high confidence extra tag; covers secondary woody structure missed; model disagreement; covers tag fruit', array['flower_fruit_bud_ambiguity','fruit_state_view_confusion','high_confidence_extra_tag','secondary_woody_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/54744', 57, 'covers leaf view confusion; covers whole plant scale split; model disagreement; covers tag leaf; covers tag twig; covers tag whole plant', array['leaf_view_confusion','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/50474', 58, 'high confidence error; covers high confidence extra tag; model disagreement; covers tag twig', array['bark_scale_confusion','high_confidence_extra_tag']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/11509', 59, 'covers whole plant scale split; model disagreement; covers tag leaf; covers tag whole plant', array['whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/11646', 60, 'covers leaf view confusion; model disagreement; covers tag leaf; covers tag twig', array['leaf_view_confusion']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/22882', 61, 'covers secondary woody structure missed; covers tag twig', array['secondary_woody_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/35324', 62, 'covers leaf view confusion; model disagreement; covers tag leaf', array['leaf_view_confusion']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/51736', 63, 'covers flower fruit bud ambiguity; model disagreement; covers tag leaf; covers tag twig', array['flower_fruit_bud_ambiguity']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/12318', 64, 'high confidence error; covers high confidence extra tag; covers tag leaf; covers tag whole plant', array['bark_scale_confusion','high_confidence_extra_tag']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/43400', 65, 'easy case; high confidence error; covers high confidence extra tag; covers tag leaf; covers tag whole plant', array['high_confidence_extra_tag']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/32255', 66, 'bioimages mismatch; gemma disagreement; covers flower fruit bud ambiguity; covers reproductive structure missed; covers whole plant scale split; model disagreement; covers tag flower; covers tag twig; covers tag whole plant', array['bioimages_single_label_ambiguity','flower_fruit_bud_ambiguity','reproductive_structure_missed','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/26141', 67, 'high confidence error; covers fruit state view confusion; covers high confidence extra tag; covers secondary woody structure missed; covers tag flower; covers tag fruit; covers tag twig', array['fruit_state_view_confusion','high_confidence_extra_tag','secondary_woody_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'kaufmannm/ke063', 68, 'gemma disagreement; covers leaf view confusion; covers secondary woody structure missed; covers tag leaf; covers tag twig; covers tag whole plant', array['leaf_view_confusion','secondary_woody_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/11515', 69, 'covers whole plant scale split; model disagreement; covers tag bark; covers tag leaf; covers tag whole plant', array['whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/10935', 70, 'covers strong model disagreement; model disagreement; covers tag leaf', array['strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/90797a', 71, 'covers strong model disagreement; covers whole plant scale split; model disagreement; covers tag leaf; covers tag whole plant', array['strong_model_disagreement','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/35725', 72, 'easy case; covers tag leaf', array[]::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/26510', 73, 'gemma disagreement; covers strong model disagreement; model disagreement; covers tag leaf', array['strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/12251', 74, 'high confidence error; covers high confidence extra tag; covers tag bark; covers tag leaf; covers tag whole plant', array['high_confidence_extra_tag']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/55748', 75, 'covers flower fruit bud ambiguity; covers fruit state view confusion; covers reproductive structure missed; model disagreement; covers tag fruit; covers tag twig', array['flower_fruit_bud_ambiguity','fruit_state_view_confusion','reproductive_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/11534', 76, 'bioimages mismatch; covers secondary woody structure missed; covers strong model disagreement; covers whole plant scale split; model disagreement; covers tag bark; covers tag leaf; covers tag twig; covers tag whole plant', array['bioimages_single_label_ambiguity','secondary_woody_structure_missed','strong_model_disagreement','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/39818', 77, 'covers flower fruit bud ambiguity; covers reproductive structure missed; model disagreement; covers tag flower; covers tag leaf', array['flower_fruit_bud_ambiguity','reproductive_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/12171', 78, 'covers whole plant scale split; model disagreement; covers tag flower; covers tag leaf; covers tag whole plant', array['whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/35303', 79, 'covers leaf view confusion; model disagreement; covers tag leaf; covers tag twig', array['leaf_view_confusion']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/10676', 80, 'high confidence error; covers high confidence extra tag; covers secondary woody structure missed; model disagreement; covers tag flower; covers tag leaf', array['high_confidence_extra_tag','secondary_woody_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/43541', 81, 'high confidence error; covers high confidence extra tag; covers tag bark', array['high_confidence_extra_tag']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/12005', 82, 'gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers fruit state view confusion; covers high confidence extra tag; covers reproductive structure missed; covers secondary woody structure missed; covers strong model disagreement; model disagreement; covers tag fruit; covers tag twig', array['flower_fruit_bud_ambiguity','fruit_state_view_confusion','high_confidence_extra_tag','reproductive_structure_missed','secondary_woody_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'bassettst/sb560', 83, 'covers whole plant scale split; model disagreement; covers tag bark; covers tag leaf; covers tag whole plant', array['bark_scale_confusion','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'kaufmannm/ke178', 84, 'gemma disagreement; covers leaf view confusion; model disagreement; rare class; covers tag leaf; covers tag twig; covers tag whole plant', array['leaf_view_confusion']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'kaufmannm/ke128', 85, 'gemma disagreement; covers whole plant scale split; model disagreement; rare class; covers tag whole plant', array['whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/10554', 86, 'bioimages mismatch; gemma disagreement; covers flower fruit bud ambiguity; covers reproductive structure missed; covers secondary woody structure missed; covers strong model disagreement; model disagreement; covers tag cone; covers tag flower; covers tag leaf; covers tag twig', array['cone_seed_under_detection','flower_fruit_bud_ambiguity','reproductive_structure_missed','secondary_woody_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/10794', 87, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers high confidence extra tag; covers reproductive structure missed; covers strong model disagreement; model disagreement; rare class; covers tag cone; covers tag flower; covers tag leaf', array['cone_seed_under_detection','flower_fruit_bud_ambiguity','high_confidence_extra_tag','reproductive_structure_missed','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/10392', 88, 'high confidence error; covers flower fruit bud ambiguity; covers high confidence extra tag; covers strong model disagreement; model disagreement; covers tag flower; covers tag leaf; covers tag twig', array['flower_fruit_bud_ambiguity','high_confidence_extra_tag','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/19587', 89, 'high confidence error; covers flower fruit bud ambiguity; covers high confidence extra tag; covers strong model disagreement; model disagreement; covers tag flower; covers tag leaf', array['flower_fruit_bud_ambiguity','high_confidence_extra_tag','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'thomas/0440-04-07', 90, 'bioimages mismatch; gemma disagreement; high confidence error; covers flower fruit bud ambiguity; covers fruit state view confusion; covers high confidence extra tag; covers strong model disagreement; model disagreement; covers tag flower; covers tag fruit', array['flower_fruit_bud_ambiguity','fruit_state_view_confusion','high_confidence_extra_tag','strong_model_disagreement']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/52483', 91, 'high confidence error; covers high confidence extra tag; covers leaf view confusion; covers tag leaf', array['high_confidence_extra_tag','leaf_view_confusion']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/37287', 92, 'easy case; high confidence error; covers high confidence extra tag; covers tag bark', array['high_confidence_extra_tag']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/12585', 93, 'model disagreement; covers tag leaf; covers tag twig', array[]::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/90876', 94, 'covers whole plant scale split; model disagreement; covers tag leaf; covers tag whole plant', array['bark_scale_confusion','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/15235', 95, 'gemma disagreement; high confidence error; covers high confidence extra tag; model disagreement; covers tag leaf', array['high_confidence_extra_tag']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/51402', 96, 'bioimages mismatch; gemma disagreement; covers flower fruit bud ambiguity; covers fruit state view confusion; covers reproductive structure missed; model disagreement; covers tag flower; covers tag leaf', array['bioimages_single_label_ambiguity','flower_fruit_bud_ambiguity','fruit_state_view_confusion','reproductive_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/13588', 97, 'high confidence error; covers high confidence extra tag; covers secondary woody structure missed; model disagreement; covers tag bark', array['bark_scale_confusion','high_confidence_extra_tag','secondary_woody_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/16741', 98, 'bioimages mismatch; high confidence error; covers high confidence extra tag; covers whole plant scale split; model disagreement; covers tag leaf; covers tag whole plant', array['high_confidence_extra_tag','whole_plant_scale_split']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'kirchoff/em2414', 99, 'gemma disagreement; high confidence error; covers high confidence extra tag; covers leaf view confusion; covers secondary woody structure missed; covers tag leaf; covers tag twig', array['high_confidence_extra_tag','leaf_view_confusion','secondary_woody_structure_missed']::text[]
),
(
  '00000000-0000-4000-8000-000000000001', 'baskauf/10674', 100, 'bioimages mismatch; covers flower fruit bud ambiguity; covers fruit state view confusion; covers reproductive structure missed; model disagreement; covers tag flower; covers tag fruit; covers tag leaf', array['flower_fruit_bud_ambiguity','fruit_state_view_confusion','reproductive_structure_missed']::text[]
)
on conflict (batch_id, image_id) do update set
  display_order = excluded.display_order,
  selection_reason = excluded.selection_reason,
  error_modes = excluded.error_modes;
