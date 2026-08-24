package com.example.llama

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

data class AgentEventRow(
    val id: String,
    val event: AgentClientEvent,
    var expanded: Boolean = false,
)

class AgentEventAdapter(
    private val events: MutableList<AgentEventRow>,
) : RecyclerView.Adapter<AgentEventAdapter.EventViewHolder>() {
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): EventViewHolder =
        EventViewHolder(LayoutInflater.from(parent.context)
            .inflate(R.layout.item_agent_event, parent, false))

    override fun onBindViewHolder(holder: EventViewHolder, position: Int) {
        val row = events[position]
        holder.type.text = row.event.type.ifBlank { "event" }
        holder.detail.text = row.event.detail
        holder.json.text = row.event.json
        holder.json.visibility = if (row.expanded) View.VISIBLE else View.GONE
        holder.toggle.text = if (row.expanded) "−" else "+"
        holder.toggle.setOnClickListener {
            row.expanded = !row.expanded
            notifyItemChanged(position)
        }
    }

    override fun getItemCount(): Int = events.size

    class EventViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val toggle: Button = view.findViewById(R.id.event_toggle)
        val type: TextView = view.findViewById(R.id.event_type)
        val detail: TextView = view.findViewById(R.id.event_detail)
        val json: TextView = view.findViewById(R.id.event_json)
    }
}
