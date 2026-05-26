export const AGENTS = [
  { id:5, name:'Grant', role:'Writing Tutor', cat:'Education', spec:'Writing & Composition', price:19, rating:4.9, reviews:5632, sessions:'45k+', c1:'#0891b2', c2:'#22d3ee', img:'grant_icon.png',
    bio:'Grant adapts to your writing level and goals — from high school essays to university dissertations, personal statements to professional reports. Grant has helped 5,600+ students find their voice, sharpen their arguments, and write with clarity and confidence. Whatever the assignment, whatever the deadline, Grant meets you exactly where you are.',
    greet:"Hello! I'm Grant, your writing tutor. What are we working on today? Share your draft, your prompt, or just tell me where you're stuck.",
    tags:['Essays','Research Papers','Creative Writing','Grammar'],
    canHelp:['Essay structure, thesis development & argumentation','Grammar, style, clarity & sentence-level editing','Research papers: sourcing, citations & academic tone','Personal statements, college essays & cover letters','Creative writing: fiction, poetry & narrative craft','Test prep: SAT/ACT writing, AP Language & AP Literature'],
    expect:"Grant meets you exactly where you are. Expect honest, specific feedback — not just encouragement. You'll get line-level edits, structural suggestions, and clear explanations of why changes make your writing stronger. Learning at your pace, not Grant's." },
]

export const REVIEWS = {
  5:[
    { name:'Sophie L.', date:'March 2026',    rating:5, text:"I submitted my college essay to 12 schools and got into my top choice. Grant helped me find the real story I was trying to tell and cut everything that wasn't it. Total game changer." },
    { name:'Omar A.',   date:'February 2026', rating:5, text:"Used Grant to prep for the AP Language exam. The feedback on my rhetorical analysis essays was more detailed than anything my teacher gave me. Scored a 5." },
    { name:'Chloe W.',  date:'January 2026',  rating:5, text:"My daughter's English grade went from a C to an A- in one semester. Grant doesn't just fix her writing — it explains why each change makes it better. She's actually learning." },
    { name:'Ben S.',    date:'December 2025', rating:4, text:"Solid for everything writing-related. The thesis and structure feedback is genuinely excellent. Best writing tutor I've had — human or AI. Worth every penny of the $19." },
  ],
}

export const AGENTS_COMING_SOON = [
  { id:101, name:'Aria',  role:'Therapist',        cat:'Mental Health', spec:'Anxiety & CBT',    price:29, rating:4.9, reviews:0, sessions:'—', c1:'#7c3aed', c2:'#a78bfa', img:'aria_icon.png', comingSoon:true,
    bio:'', greet:'', tags:[], canHelp:[], expect:'' },
  { id:102, name:'Lex',   role:'Lawyer',            cat:'Legal',         spec:'Contract & IP',   price:49, rating:4.8, reviews:0, sessions:'—', c1:'#1d4ed8', c2:'#60a5fa', img:'lex_icon.png', comingSoon:true,
    bio:'', greet:'', tags:[], canHelp:[], expect:'' },
  { id:103, name:'Max',   role:'Financial Advisor', cat:'Finance',       spec:'Portfolio & Tax', price:39, rating:4.7, reviews:0, sessions:'—', c1:'#059669', c2:'#34d399', img:'max_icon.png', comingSoon:true,
    bio:'', greet:'', tags:[], canHelp:[], expect:'' },
]

export const ALL_AGENTS = [...AGENTS, ...AGENTS_COMING_SOON]

export const CATS = ['All', 'Education', 'Mental Health', 'Legal', 'Finance']

export const REPLIES = [
  "Good start — let me show you a few specific changes that will make this much stronger.",
  "The core idea is solid. The issue is in how you're structuring the argument. Here's what I'd do.",
  "Your thesis is doing too much work. Let's narrow it down so every paragraph earns its place.",
  "Let's look at this sentence by sentence — I'll show you exactly where the clarity breaks down.",
]
