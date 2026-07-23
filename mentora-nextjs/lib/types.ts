// Mirrors the FastAPI Pydantic / mock response models exactly (see
// backend/routers/*.py). Keep these in sync — if a mock JSON shape
// changes, update the matching type here in the same commit.

export interface UserPublic {
  id: number;
  first_name: string;
  last_name: string;
  email: string;
  university: string | null;
  exam_goal: string | null;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: UserPublic;
}

// --- dashboard -------------------------------------------------------

export interface DashboardStats {
  exam_sessions: number;
  avg_score: number;
  avg_score_delta: number;
  videos_watched: number;
  videos_watched_this_week: number;
  top_career_match: { title: string; fit_percent: number };
}

export interface ScoreTrendPoint {
  session: string;
  score: number;
}

export interface WeeklySnapshotDay {
  day: string;
  topic: string;
  status: "done" | "today" | "missed" | "pending";
}

export interface WeakTopic {
  subject: string;
  topic: string;
  score: number;
  note: string | null;
}

export interface NotificationItem {
  id: number;
  message: string;
  is_read: boolean;
}

// --- exam --------------------------------------------------------------

export interface TimetableRow {
  day: string;
  topic: string;
  subject: string;
  duration: string;
  status: "pending" | "done" | "today" | "missed";
}

export interface Timetable {
  days_to_exam: number;
  rows: TimetableRow[];
}

export interface TodayTopic {
  topic: string;
  explainer: string;
  key_points: string[];
  videos: { title: string; source: string; duration_min: number }[];
}

export interface MCQOption {
  value: string;
  label: string;
}

export interface MCQQuestion {
  question_id: string;
  question: string;
  options: MCQOption[];
  correct_answer: string;
  total_in_set: number;
  remaining_in_set: number;
  generated_by: string;
}

export interface MCQSubmitResult {
  correct: boolean;
  correct_answer: string;
  weak_topics_flagged: string[];
}

// --- career --------------------------------------------------------------

export interface CVExtracted {
  filename: string;
  gpa: number;
  technical_skills: string[];
  certifications: string[];
  projects: string[];
}

export interface CareerRankingRow {
  career_id: number;
  title: string;
  match_score: number;
  skill_gap: string;
}

export interface RoadmapStep {
  step: string;
  title: string;
  subtitle: string;
  status: "completed" | "in_progress" | "pending";
}

export interface Roadmap {
  career_id: number;
  career_title: string;
  steps: RoadmapStep[];
}

export interface InterviewStartResult {
  interview_id: string;
  first_question: string;
}

export interface InterviewAskResult {
  answer: string;
}

// --- learning portal -----------------------------------------------------

export interface LearningStats {
  videos_in_playlist: number;
  videos_watched_this_week: number;
  mcq_assignments: number;
  mcq_pending_today: number;
  course_pdfs: number;
  course_pdfs_downloaded: number;
  avg_mcq_score: number;
  avg_mcq_score_delta: number;
}

export interface PlaylistVideo {
  id: number;
  title: string;
  source: string;
  duration_min: number;
  subject: string;
  watched: boolean;
}

export interface Assignment {
  id: number;
  title: string;
  subject: string;
  status: "pending" | "done";
  due: string;
}

export interface Material {
  id: number;
  title: string;
  subject: string;
  downloaded: boolean;
}
