import type { CodeExerciseDef } from './types'

const modules = import.meta.glob<{ default: CodeExerciseDef[] }>(['./*.ts', '!./index.ts', '!./types.ts', '!./*.test.ts'], { eager: true })

export const CODE_EXERCISES: CodeExerciseDef[] = Object.values(modules).flatMap((m) => m.default)
export const codeExerciseById = (id: string) => CODE_EXERCISES.find((e) => e.id === id)
export type { CodeExerciseDef }
