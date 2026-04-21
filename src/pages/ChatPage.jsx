import { useReducer, useCallback, useState } from 'react'
import Sidebar from '../components/Sidebar'
import ChatThread from '../components/ChatThread'
import Composer from '../components/Composer'
import UpgradeTab from '../components/UpgradeTab'
import UploadToast from '../components/UploadToast'
import { useUpload } from '../hooks/useUpload'
import './ChatPage.css'

const CANNED_REPLIES = [
  `Sure, I can help you get started with creating a chatbot using GPT in Python. Here are the basic steps you'll need to follow:

1. **Install the required libraries:** You'll need to install the transformers library from Hugging Face to use GPT. You can install it using pip.

2. **Load the pre-trained model:** GPT comes in several sizes and versions, so you'll need to choose the one that fits your needs. You can load a pre-trained GPT model. This loads the 1.3B parameter version of GPT-Neo, which is a powerful and relatively recent model.

3. **Create a chatbot loop:** You'll need to create a loop that takes user input, generates a response using the GPT model, and outputs it to the user. Here's an example loop that uses the input() function to get user input and the gpt() function to generate a response.

4. **Add some personality to the chatbot:** While GPT can generate text, it doesn't have any inherent personality or style. You can make your chatbot more interesting by adding custom prompts or responses that reflect your desired personality.

These are just the basic steps to get started with a GPT chatbot in Python. Depending on your requirements, you may need to add more features or complexity to the chatbot. Good luck!`,
  `Chatbots can be used for a wide range of purposes, including:

**Customer service:** Chatbots can handle frequently asked questions, provide basic support, and help customers resolve issues without human intervention.

**E-commerce:** Chatbots can assist users in finding products, tracking orders, and completing purchases within a conversational interface.

**Healthcare:** Medical chatbots can provide symptom checking, appointment scheduling, and medication reminders to patients.

**Education:** Chatbots can deliver personalized learning experiences, answer student questions, and provide tutoring support around the clock.`,
  `That's a great question! Here's what you should know:

The key difference lies in how each approach handles state and side effects. When building scalable systems, it's generally better to start with a simple architecture and only introduce complexity when the need arises.

A few things to keep in mind:
1. Start with the simplest solution that works
2. Measure before optimizing
3. Document your architectural decisions
4. Review and refactor regularly`,
]

let nextId = 1
function makeId() { return nextId++ }

const SEED_MESSAGES = [
  { id: makeId(), role: 'user', content: 'Create a chatbot gpt using python language what will be step for that' },
  { id: makeId(), role: 'assistant', content: CANNED_REPLIES[0] },
  { id: makeId(), role: 'user', content: 'What is use of that chatbot ?' },
  { id: makeId(), role: 'assistant', content: CANNED_REPLIES[1] },
]

function messagesReducer(state, action) {
  switch (action.type) {
    case 'ADD_MESSAGE':
      return { ...state, messages: [...state.messages, action.payload], pending: false }
    case 'SET_PENDING':
      return { ...state, pending: true }
    default:
      return state
  }
}

export default function ChatPage() {
  const [state, dispatch] = useReducer(messagesReducer, {
    messages: SEED_MESSAGES,
    pending: false,
  })
  const [showToast, setShowToast] = useState(false)

  const upload = useUpload({
    onComplete: () => setShowToast(true),
  })

  const handleSend = useCallback((text) => {
    const userMsg = { id: makeId(), role: 'user', content: text }
    dispatch({ type: 'ADD_MESSAGE', payload: userMsg })
    dispatch({ type: 'SET_PENDING' })
    const reply = CANNED_REPLIES[Math.floor(Math.random() * CANNED_REPLIES.length)]
    setTimeout(() => {
      dispatch({ type: 'ADD_MESSAGE', payload: { id: makeId(), role: 'assistant', content: reply } })
    }, 500)
  }, [])

  return (
    <div className="chat-page-outer">
      <div className="chat-page-card">
        <Sidebar activeTitle="Create Chatbot GPT..." />
        <div className="chat-main">
          <ChatThread messages={state.messages} pending={state.pending} />
          <Composer
            onSend={handleSend}
            onFileSelect={upload.uploadFile}
            uploadStatus={upload.status}
            disabled={state.pending}
          />
        </div>
        <div className="chat-page-upgrade">
          <UpgradeTab />
        </div>
      </div>
      {showToast && (
        <UploadToast
          status={upload.status}
          fileName={upload.fileName}
          error={upload.error}
          onDismiss={() => {
            setShowToast(false)
            upload.reset()
          }}
        />
      )}
    </div>
  )
}
